"""Regression checks for borderless tables and figure crops."""
from pathlib import Path
import fitz
from backend.src.document_pipeline import _caption_candidates, _caption_numeric_table, _figure_bounds

SAMPLE = Path(__file__).with_name('sample_research_paper.pdf')


def test_caption_numeric_table_detects_sample_metrics():
    with fitz.open(SAMPLE) as doc:
        page = doc[4]
        caps = _caption_candidates(page, 'table')
        assert caps
        table = _caption_numeric_table(page, caps[0])
        assert table is not None
        cells = str(table['rows'])
        assert 'Gemini 1.5 Pro' in cells
        assert '0.935' in cells


def test_figure_bounds_excludes_caption():
    with fitz.open(SAMPLE) as doc:
        page = doc[2]
        caps = _caption_candidates(page, 'figure')
        assert caps
        clip = _figure_bounds(page, caps[0])
        assert clip is not None
        assert clip.y1 < caps[0]['rect'].y0
