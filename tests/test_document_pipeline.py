from backend.src.document_pipeline import _table_to_markdown, _sections_from_pages


def test_table_to_markdown_preserves_cells():
    md = _table_to_markdown([["Model", "Accuracy"], ["A", "91.2%"], ["B", "94.6%"]])
    assert "Model" in md and "Accuracy" in md and "94.6%" in md


def test_sections_fall_back_to_full_paper():
    sections = _sections_from_pages("Some paper text without known headings.")
    assert "Full Paper" in sections


def test_sections_detect_common_headings():
    text = "Abstract\nThis is a sufficiently long abstract with details.\n\nIntroduction\nThis is a sufficiently long introduction with detail."
    sections = _sections_from_pages(text)
    assert any("Abstract" in key for key in sections)
