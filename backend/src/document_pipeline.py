"""Extract text, tables and figures while preserving page-level provenance."""
from __future__ import annotations

import base64
import os
import re
from pathlib import Path
from typing import Any

import fitz
import pdfplumber

from backend.src.llm import LLMService


def _clean_text(text: str | None) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _table_to_markdown(rows: list[list[Any]]) -> str:
    cleaned = [[_clean_text(str(cell or "")).replace("|", "\\|").replace("\n", " ") for cell in row] for row in rows]
    cleaned = [row for row in cleaned if any(cell for cell in row)]
    if not cleaned:
        return ""
    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]
    header = cleaned[0]
    separator = ["---"] * width
    body = cleaned[1:] or []
    return "| " + " | ".join(header) + " |\n| " + " | ".join(separator) + " |" + ("\n" + "\n".join("| " + " | ".join(row) + " |" for row in body) if body else "")


def _find_caption(page_text: str, label: str, index: int) -> str:
    pattern = re.compile(rf"\b{label}\s*{index}\s*[:.\-–]?\s*([^\n]{{0,220}})", re.IGNORECASE)
    match = pattern.search(page_text or "")
    return (f"{label.title()} {index}: {match.group(1).strip()}" if match and match.group(1).strip() else f"{label.title()} {index}")


def _vision_description(image_path: Path, llm_service: LLMService) -> str:
    """Optional image understanding through Groq's vision endpoint; gracefully degrades."""
    model = os.getenv("GROQ_VISION_MODEL", "").strip()
    if not model:
        return ""
    try:
        return llm_service.describe_image(image_path, model=model)
    except Exception:
        # A figure should not make the entire PDF upload fail when vision is unavailable.
        return ""


def _caption_candidates(page, kind: str) -> list[dict]:
    """Read captions with coordinates so figures/tables match the same page."""
    pat = re.compile(r"^\s*(?:fig(?:ure)?\.?|table)\s*[\.:#-]?\s*(\d+[a-z]?)\s*[\.:)–-]?\s*(.*)", re.I)
    result = []
    for block in page.get_text("blocks"):
        if len(block) < 5:
            continue
        for line in block[4].splitlines():
            match = pat.match(line.strip())
            if not match:
                continue
            is_figure = line.strip().lower().startswith("fig")
            if is_figure != (kind == "figure"):
                continue
            result.append({"number": match.group(1), "text": line.strip()[:260], "rect": fitz.Rect(block[:4])})
    return result


def _paper_metadata(page, document_title: str) -> dict:
    """Heuristic bibliography extraction; authors are marked unknown if uncertain."""
    lines = [line.strip() for line in page.get_text("text").splitlines() if line.strip()]
    abstract_match = re.search(r"\babstract\b[\s:—.-]*(.*?)(?=\n\s*(?:keywords|index terms|(?:1|i)[.\s]+introduction)\b|\n\s*introduction\b|$)", page.get_text("text"), re.I | re.S)
    abstract = _clean_text(abstract_match.group(1))[:3500] if abstract_match else ""
    if not abstract:
        match = re.search(r"(?is)\babstract\s*[—:.-]?\s*(.{80,2200}?)(?:\bkeywords\b|\bintroduction\b)",page.get_text("text"))
        abstract = _clean_text(match.group(1)) if match else ""
    blocks = []
    for block in page.get_text("dict").get("blocks", []):
        if "lines" not in block:
            continue
        for line in block["lines"]:
            spans=line.get("spans", [])
            text = "".join(x.get("text", "") for x in spans).strip()
            if text and spans:
                blocks.append((max(float(x.get("size", 0)) for x in spans), line["bbox"][1], text, line["bbox"][3]))
    top = [b for b in blocks if b[1] < page.rect.height * .42 and len(b[2]) > 8 and not re.search(r"arxiv|https?://|doi:|@|university|department",b[2],re.I)]
    if top:
        largest=max(x[0] for x in top)
        title_lines=[x for x in top if x[0]>=largest*.91 and x[1]<page.rect.height*.36]
        
        # PDF title often uses letter-styled spans (S + UFFICIENT). Joining
        # extracted spans with spaces corrupts words. Read actual PDF words
        # at the detected large-font title line coordinates instead.
        word_rows = page.get_text("words", sort=True)
        reconstructed = []
        for title_line in sorted(title_lines, key=lambda a: a[1]):
            y0, y1 = title_line[1], title_line[3]
            line_words = [w for w in word_rows if y0 - 2 <= (w[1] + w[3]) / 2 <= y1 + 2]
            line_words.sort(key=lambda w: (w[5], w[6], w[7]))
            line_text = " ".join(w[4] for w in line_words)
            if line_text and line_text not in reconstructed:
                reconstructed.append(line_text)
        candidate = " ".join(reconstructed)[:350] if reconstructed else " ".join(x[2] for x in title_lines)[:350]
        candidate = re.sub(r"\s+", " ", candidate).strip()
        if len(candidate)>12 and candidate.lower() not in {"abstract", "introduction"}:
            document_title=candidate
    # Author lines are very format-dependent. Best effort from just below title.
    authors=""
    if top:
        ordered=sorted(top,key=lambda a:a[1])
        max_size=max(x[0] for x in ordered)
        title_bottom=max((x[1] for x in ordered if x[0]>=max_size*.91),default=0)
        after=[x[2] for x in ordered if title_bottom < x[1] < title_bottom+95 and x[0]<max_size*.91]
        after=[a for a in after if not re.search(r"abstract|university|department|@|http|arxiv|school of|institute|faculty", a,re.I)]
        if after: authors="; ".join(after[:3])[:300]
    return {"title":document_title,"authors":authors,"abstract":abstract}


def _table_candidates(plumber_page):
    """Find bordered and borderless tables without accepting arbitrary paragraphs."""
    settings = [
        {"vertical_strategy": "lines", "horizontal_strategy": "lines",
         "snap_tolerance": 5, "join_tolerance": 5, "intersection_tolerance": 5},
        {"vertical_strategy": "lines_strict", "horizontal_strategy": "lines_strict",
         "snap_tolerance": 5, "join_tolerance": 5},
        {"vertical_strategy": "text", "horizontal_strategy": "text",
         "min_words_vertical": 2, "min_words_horizontal": 2,
         "text_x_tolerance": 2, "text_y_tolerance": 3,
         "intersection_tolerance": 9, "snap_tolerance": 5},
    ]
    found = []
    for mode, setting in enumerate(settings):
        # Restrict text-mode table discovery to a moderate-height region.
        # Otherwise pdfplumber may turn an entire prose page into a fake table.
        try:
            detected = plumber_page.find_tables(table_settings=setting)
        except Exception:
            continue
        for table in detected:
            if mode == 2 and (table.bbox[3] - table.bbox[1]) > plumber_page.height * .40:
                continue
            rows = table.extract(x_tolerance=2, y_tolerance=3) or []
            rows = [[_clean_text(str(cell or "")) for cell in row] for row in rows]
            if len(rows) < 2 or max(map(len, rows), default=0) < 2:
                continue
            filled = sum(bool(cell) for row in rows for cell in row)
            total = sum(len(row) for row in rows)
            if filled < 4 or filled / max(1, total) < .36:
                continue
            # A text-mode detection has to exhibit repeated populated columns.
            occupied_rows = sum(sum(bool(c) for c in row) >= 2 for row in rows)
            if mode == 2 and (occupied_rows < 3 or table.bbox[3] - table.bbox[1] < 28):
                continue
            score = filled + occupied_rows * 2 + (2 if mode < 2 else 0)
            bbox = table.bbox
            overlapping = next((old for old in found if _bbox_overlap_ratio(bbox, old["bbox"]) > .55), None)
            if overlapping:
                if score > overlapping["score"]:
                    overlapping.update(rows=rows, score=score, bbox=bbox, bordered=(mode < 2))
            else:
                found.append({"bbox": bbox, "rows": rows, "score": score, "bordered": mode < 2})
    return found


def _near_table_caption(table_bbox, captions, page_height):
    """Match nearest table caption above or below the table without a global index."""
    if not captions:
        return None
    center = (table_bbox[1] + table_bbox[3]) / 2
    nearest = min(captions, key=lambda c: min(abs(c["rect"].y0 - table_bbox[3]),
                                                abs(c["rect"].y1 - table_bbox[1])))
    gap = min(abs(nearest["rect"].y0 - table_bbox[3]),
              abs(nearest["rect"].y1 - table_bbox[1]))
    return nearest if gap < min(130, page_height * .18) else None


def _figure_bounds(page, caption):
    """Crop rendered figure graphics rather than a fixed-height paragraph region.

    Both embedded images and PDF vector drawings contribute bounds. When no
    graphical region is found, return None rather than a misleading text crop.
    """
    cap = caption["rect"]
    width = page.rect.width
    search_top = max(12, cap.y0 - min(360, page.rect.height * .48))
    graphics = []
    for d in page.get_drawings():
        r = fitz.Rect(d["rect"])
        # Very small strokes are usually text decoration or separators.
        if r.is_empty or r.y1 > cap.y0 - 2 or r.y0 < search_top:
            continue
        if r.width < 8 and r.height < 8:
            continue
        if r.width > width * .97 and r.height < 3:
            continue
        graphics.append(r)
    for img in page.get_images(full=True):
        for r in page.get_image_rects(img[0]):
            if r.y0 >= search_top and r.y1 <= cap.y0 + 1 and r.width > 40 and r.height > 25:
                graphics.append(r)
    # Image blocks can capture inline images excluded from get_images().
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") == 1:
            r = fitz.Rect(block["bbox"])
            if r.y0 >= search_top and r.y1 <= cap.y0 + 1 and r.width > 40 and r.height > 25:
                graphics.append(r)
    if not graphics:
        return None
    # Work upward from the graphics closest to the caption. Join vertically
    # neighboring graphics, avoiding figures higher up on the same page.
    graphics.sort(key=lambda r: r.y1, reverse=True)
    chosen = [graphics[0]]
    top = graphics[0].y0
    for r in graphics[1:]:
        if r.y1 >= top - 22:
            chosen.append(r)
            top = min(top, r.y0)
    rect = fitz.Rect(min(r.x0 for r in chosen), min(r.y0 for r in chosen),
                     max(r.x1 for r in chosen), max(r.y1 for r in chosen))
    if rect.width < 90 or rect.height < 35:
        return None
    # Keep the crop within the graphics footprint so neighboring prose is not
    # captured. A few pixels of padding avoids cutting diagram lines.
    rect = fitz.Rect(max(6, rect.x0 - 7), max(search_top, rect.y0 - 7),
                     min(width - 6, rect.x1 + 7), min(cap.y0 - 3, rect.y1 + 7))
    return rect if rect.width >= 90 and rect.height >= 35 else None


def _caption_numeric_table(page, caption):
    """Recover common borderless scientific tables using aligned numeric columns.

    Conservative fallback: requires several neighboring data rows with at least
    two aligned numerical values. Never treat an arbitrary prose paragraph as a
    numeric table. More complex tables still need a document-layout model.
    """
    words = page.get_text("words", sort=True)
    cap_y = caption["rect"].y1
    region = [w for w in words if cap_y < w[1] < min(page.rect.height - 35, cap_y + 300)]
    groups = []
    for w in sorted(region, key=lambda a: (a[1], a[0])):
        cy = (w[1] + w[3]) / 2
        if not groups or abs(groups[-1][0] - cy) > 4:
            groups.append([cy, [w]])
        else:
            groups[-1][1].append(w)
    number_re = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)%?$|^[✓✗✔]$")
    best = []
    current = []
    previous_y = None
    for y, row_words in groups:
        nums = [w for w in row_words if number_re.match(w[4])]
        qualifies = len(nums) >= 2 and len(row_words) <= 18
        if qualifies and (previous_y is None or y - previous_y < 21):
            current.append((y, row_words, nums))
        elif qualifies:
            if len(current) > len(best): best = current
            current = [(y, row_words, nums)]
        else:
            if len(current) > len(best): best = current
            current = []
        previous_y = y if qualifies else None
    if len(current) > len(best): best = current
    if len(best) < 3:
        return None
    centers = []
    for _, _, nums in best:
        for w in nums:
            x = w[0]
            cluster = next((c for c in centers if abs(c[0] - x) < 11), None)
            if cluster:
                n = cluster[1]
                cluster[0] = (cluster[0] * n + x) / (n + 1)
                cluster[1] += 1
            else:
                centers.append([x, 1])
    columns = sorted(c[0] for c in centers if c[1] >= max(3, len(best) * .65))
    if len(columns) < 2:
        return None
    first_x = columns[0]
    rows = []
    for _, row_words, _ in best:
        cells = [[] for _ in range(len(columns) + 1)]
        for w in sorted(row_words, key=lambda v: v[0]):
            x = w[0]
            if x < first_x - 10:
                cells[0].append(w[4])
            else:
                index = min(range(len(columns)), key=lambda i: abs(columns[i] - x))
                cells[index + 1].append(w[4])
        if not cells[0] or sum(bool(c) for c in cells[1:]) < 2:
            continue
        rows.append([" ".join(c) for c in cells])
    if len(rows) < 3:
        return None
    # Header labels can vary wildly, so use explicit placeholders rather than
    # fabricate metric names. Preserve exactly extracted numeric values.
    headers = ["Method / Item"] + [f"Value {i}" for i in range(1, len(columns) + 1)]
    return {"rows": [headers] + rows, "bbox": (min(w[0] for _, rw, _ in best for w in rw),
            best[0][0] - 8, max(w[2] for _, rw, _ in best for w in rw), best[-1][0] + 8),
            "score": len(rows) * len(columns), "bordered": False}


def _bbox_overlap_ratio(a,b):
    dx=max(0,min(a[2],b[2])-max(a[0],b[0]));dy=max(0,min(a[3],b[3])-max(a[1],b[1]))
    ar=lambda v:max(1,(v[2]-v[0])*(v[3]-v[1]))
    return dx*dy/min(ar(a),ar(b))


def process_pdf(pdf_path: Path, paper_id: str, original_name: str, figure_dir: Path) -> dict[str, Any]:
    pages=[];page_texts=[];figures=[];tables=[]
    paper_fig_dir=figure_dir/paper_id;paper_fig_dir.mkdir(parents=True,exist_ok=True)
    vision=LLMService();title=Path(original_name).stem.replace("_"," ")
    with fitz.open(pdf_path) as pdf:
        page_count=len(pdf)
        # PDF metadata is frequently a stale filename or generic export title.
        reported=(pdf.metadata or {}).get("title","")
        if reported and len(reported.strip())>8:title=reported.strip()
        metadata=_paper_metadata(pdf[0],title) if page_count else {"title":title,"authors":"","abstract":""}
        title=metadata["title"]
        with pdfplumber.open(pdf_path) as plumber:
            for idx,page in enumerate(pdf):
                num=idx+1;text=_clean_text(page.get_text("text",sort=True))
                page_texts.append(text);pages.append(f"[PAGE {num}]\n{text}" if text else f"[PAGE {num}]\n[No extractable text]")
                fig_captions=_caption_candidates(page,"figure")
                tab_captions=_caption_candidates(page,"table")
                table_candidates = _table_candidates(plumber.pages[idx])
                # Borderless table text often defeats whole-page detection.
                # Re-run alignment detection on a localized region near a Table caption.
                for tc in tab_captions:
                    caption_y0, caption_y1 = tc["rect"].y0, tc["rect"].y1
                    for y0, y1 in (
                        (caption_y1 + 2, min(page.rect.height - 20, caption_y1 + 280)),
                        (max(25, caption_y0 - 270), caption_y0 - 2),
                    ):
                        if y1 - y0 < 45:
                            continue
                        try:
                            cropped = plumber.pages[idx].crop((25, y0, plumber.pages[idx].width - 25, y1))
                            for item in _table_candidates(cropped):
                                if any(_bbox_overlap_ratio(item["bbox"], old["bbox"]) > .55 for old in table_candidates):
                                    continue
                                if item["bbox"][3] - item["bbox"][1] < 30:
                                    continue
                                table_candidates.append(item)
                        except Exception:
                            continue
                for tc in tab_captions:
                    aligned = _caption_numeric_table(page, tc)
                    if aligned and not any(_bbox_overlap_ratio(aligned["bbox"], x["bbox"]) > .55 for x in table_candidates):
                        table_candidates.append(aligned)
                for table in table_candidates:
                    rows=table["rows"];md=_table_to_markdown(rows)
                    if not md:continue
                    bbox=table["bbox"]
                    near = _near_table_caption(bbox, tab_captions, page.rect.height)
                    # Avoid interpreting ordinary prose / references as borderless tables.
                    # Captionless tables need actual ruling lines to be trusted.
                    if not near:
                        if not table["bordered"]:
                            continue
                        # A ruled callout or shaded text box is not a data table.
                        populated_rows = sum(sum(bool(c) for c in row) >= 2 for row in rows)
                        if populated_rows < max(3, int(len(rows) * .65)):
                            continue
                    caption = near["text"] if near else f"Table on page {num}"
                    ident=f"table_{len(tables)+1}"
                    tables.append({"table_id":ident,"page":num,"caption":caption,"markdown":md,
                     "row_count":len(rows),"column_count":max(len(r) for r in rows),
                     "content":f"{caption} (page {num})\n{md}"})
                # Render only the detected graphic footprint. Fixed-height crops
                # previously included paragraphs surrounding figures.
                for fc in fig_captions:
                    clip = _figure_bounds(page, fc)
                    if clip is None:
                        continue
                    try:
                        pix=page.get_pixmap(matrix=fitz.Matrix(1.6,1.6),clip=clip,alpha=False,colorspace=fitz.csRGB)
                        filename=f"{paper_id}_{len(figures)+1}.png";path=paper_fig_dir/filename
                        pix.save(path)
                        if path.stat().st_size<3500:path.unlink(missing_ok=True);continue
                        description=_vision_description(path,vision)
                        caption=fc["text"]
                        figures.append({"figure_id":f"figure_{len(figures)+1}","page":num,"caption":caption,
                          "description":description,"image_url":f"/api/figure/{paper_id}/{filename}","filename":filename,
                          "content":f"Figure on page {num}. Caption: {caption}. Visual description: {description or 'Image preview available; visual details not verified.'}"})
                    except Exception:continue
                # Raster fallback only if a page has no captioned figure.
                if not fig_captions:
                    seen=set()
                    for image in page.get_images(full=True):
                        xref=image[0]
                        if xref in seen:continue
                        seen.add(xref)
                        try:
                            rects=page.get_image_rects(xref)
                            if not rects:continue
                            rect=max(rects,key=lambda r:r.get_area())
                            if rect.width<110 or rect.height<80 or rect.get_area()<9000:continue
                            pix=page.get_pixmap(matrix=fitz.Matrix(1.6,1.6),clip=rect,alpha=False,colorspace=fitz.csRGB)
                            filename=f"{paper_id}_{len(figures)+1}.png";path=paper_fig_dir/filename;pix.save(path)
                            if path.stat().st_size<3500:path.unlink(missing_ok=True);continue
                            figures.append({"figure_id":f"figure_{len(figures)+1}","page":num,"caption":f"Uncaptioned image, page {num}","description":"",
                             "image_url":f"/api/figure/{paper_id}/{filename}","filename":filename,
                             "content":f"Uncaptioned image on page {num}; visually inspect preview."})
                        except Exception:continue
    full_text="\n\n".join(pages)
    sections=_sections_from_pages(full_text)
    chunks=_make_chunks(full_text,page_texts,tables,figures)
    if metadata["abstract"]:
        chunks.insert(0,{"text":f"[Paper metadata]\nTitle: {title}\nAuthors: {metadata['authors']}\nAbstract: {metadata['abstract']}","content_type":"metadata","page":1,"label":"Paper metadata"})
    return {"title":title,"authors":metadata["authors"],"abstract":metadata["abstract"],"page_count":page_count,
      "full_text":full_text,"sections":sections,"tables":tables,"figures":figures,"chunks":chunks}


def _sections_from_pages(text: str) -> dict[str, str]:
    """Lightweight heading detection with safe full-paper fallback."""
    pattern = re.compile(r"(?m)^\s*((?:\d+(?:\.\d+)*\s+)?(?:Abstract|Introduction|Related Work|Background|Methodology|Methods|Materials and Methods|Experiments?|Results|Discussion|Conclusion|Conclusions|Limitations|Future Work|References)\b[^\n]{0,120})\s*$", re.I)
    matches = list(pattern.finditer(text))
    sections: dict[str, str] = {}
    if not matches:
        return {"Full Paper": text[:60000]}
    for idx, match in enumerate(matches):
        name = re.sub(r"\s+", " ", match.group(1)).strip()
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if len(content) > 40:
            key = name
            suffix = 2
            while key in sections:
                key = f"{name} ({suffix})"
                suffix += 1
            sections[key] = content[:30000]
    return sections or {"Full Paper": text[:60000]}


def _make_chunks(full_text: str, page_texts: list[str], tables: list[dict], figures: list[dict], chunk_size: int = 1100, overlap: int = 180) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    # Page-aware text chunks keep citations meaningful.
    for page_index, text in enumerate(page_texts):
        if not text.strip():
            continue
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({"text": f"[PAGE {page_index + 1}]\n{chunk}", "paper_id": "", "content_type": "text", "page": page_index + 1, "label": f"Page {page_index + 1}"})
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
    for table in tables:
        chunks.append({"text": table["content"], "content_type": "table", "page": table["page"], "label": table["table_id"], "caption": table["caption"]})
    for figure in figures:
        content = figure.get("content", "")
        if content.strip():
            chunks.append({"text": content, "content_type": "figure", "page": figure["page"], "label": figure["figure_id"], "caption": figure["caption"]})
    return chunks
