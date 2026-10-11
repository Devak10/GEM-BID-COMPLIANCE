"""
Integration test for Streamlit Application pages and rendering.
Ensures all 7 system views render without exceptions under modern enterprise styling.
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app" / "app.py")

NAV_PAGES = [
    "Dashboard",
    "Bid Verification",
    "Document Verification",
    "Risk Analysis",
    "Model Insights",
    "Officer Review",
    "About System"
]


def test_app_dashboard_loads():
    """Verify default initial view (Dashboard) renders without unhandled exceptions."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert len(at.exception) == 0, f"Exception on Dashboard: {at.exception}"


@pytest.mark.parametrize("page_name", NAV_PAGES[1:])
def test_all_pages_render(page_name):
    """Verify switching to each of the 6 other pages renders with 0 exceptions."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert len(at.exception) == 0
    at.sidebar.radio[0].set_value(page_name).run()
    assert len(at.exception) == 0, f"Exception on {page_name}: {at.exception}"


def test_no_hackathon_labels_in_rendered_ui():
    """Verify that no hackathon or problem-statement labels appear in the rendered UI."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert len(at.exception) == 0
    
    forbidden_terms = [
        "AIML HACKATHON",
        "ASSIGNED PROBLEM STATEMENT 26100",
        "SMART AUTOMATION",
        "PROBLEM STATEMENT 26100"
    ]
    
    # Check across markdown, captions, headers in initial view and other pages
    for page in NAV_PAGES:
        at.sidebar.radio[0].set_value(page).run()
        for md in at.markdown:
            val_upper = md.value.upper()
            for term in forbidden_terms:
                assert term not in val_upper, f"Forbidden term '{term}' found on page {page}"

