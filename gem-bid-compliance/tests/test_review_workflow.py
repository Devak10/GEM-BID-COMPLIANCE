"""
AI-Powered Integrated Bid Compliance Verification Platform for GeM Procurement
AIML Hackathon Problem Statement ID: 26100
Organization: Chennai Petroleum Corporation Limited (CPCL)
Theme: Smart Automation

Unit and Integration Tests: Bid Review Workflow & Audit Trail
File: tests/test_review_workflow.py
"""

import os
import sqlite3
import pytest
from pathlib import Path

from src.review_workflow import (
    init_db,
    create_review_record,
    get_review_record,
    update_review_status,
    add_review_remark,
    get_review_history,
    calculate_review_statistics,
    validate_review_action,
    list_all_reviews,
    VALID_REVIEW_STATUSES,
    VALID_OFFICER_ACTIONS,
    WORKFLOW_DISCLAIMER
)


@pytest.fixture
def temp_db(tmp_path):
    """Provides a fresh temporary SQLite database path for isolated test execution."""
    db_file = str(tmp_path / "test_review_audit.db")
    return db_file


def test_database_initialization(temp_db):
    """1. Test that init_db initializes the database and creates required tables and indexes."""
    assert not os.path.exists(temp_db)
    init_db(temp_db)
    assert os.path.exists(temp_db)

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()

    assert "bid_reviews" in tables
    assert "review_audit_log" in tables


def test_create_review(temp_db):
    """2. Test creating a review record with AI metadata and initial status."""
    rec = create_review_record(
        bid_id="BID_TEST_001",
        tender_id="GEM/2026/B/8888",
        company_name="Alpha Tech Pvt Ltd",
        predicted_compliance="Compliant",
        compliance_probability=0.92,
        non_compliance_probability=0.08,
        risk_level="Low",
        review_priority="STANDARD REVIEW",
        officer_remarks="",
        document_completeness=95.0,
        documentation_consistency="HIGH",
        db_path=temp_db
    )

    assert rec is not None
    assert rec["bid_id"] == "BID_TEST_001"
    assert rec["company_name"] == "Alpha Tech Pvt Ltd"
    assert rec["predicted_compliance"] == "Compliant"
    assert rec["review_status"] == "NOT REVIEWED"
    assert rec["officer_action"] == "PENDING REVIEW"
    assert rec["risk_level"] == "Low"
    assert rec["review_priority"] == "STANDARD REVIEW"
    assert rec["document_completeness"] == 95.0


def test_retrieve_review(temp_db):
    """3. Test retrieving an existing review and handling a non-existent review."""
    create_review_record(
        bid_id="BID_TEST_RETRIEVE",
        company_name="Beta Solutions",
        db_path=temp_db
    )

    found = get_review_record("BID_TEST_RETRIEVE", db_path=temp_db)
    assert found is not None
    assert found["company_name"] == "Beta Solutions"

    missing = get_review_record("BID_NON_EXISTENT", db_path=temp_db)
    assert missing is None


def test_update_review_status(temp_db):
    """4. Test updating review status through valid actions and status progression."""
    create_review_record(bid_id="BID_UPDATE_01", db_path=temp_db)

    # Transition 1: NOT REVIEWED -> REVIEW STARTED (UNDER REVIEW)
    success, msg, rec = update_review_status(
        bid_id="BID_UPDATE_01",
        action="REVIEW STARTED",
        remarks="Initiating preliminary verification.",
        db_path=temp_db
    )
    assert success is True
    assert rec["review_status"] == "UNDER REVIEW"
    assert rec["officer_action"] == "REVIEW STARTED"
    assert "Initiating preliminary verification." in rec["officer_remarks"]

    # Transition 2: UNDER REVIEW -> REVIEW COMPLETED (REVIEWED)
    success2, msg2, rec2 = update_review_status(
        bid_id="BID_UPDATE_01",
        action="REVIEW COMPLETED",
        remarks="All documents verified.",
        db_path=temp_db
    )
    assert success2 is True
    assert rec2["review_status"] == "REVIEWED"
    assert rec2["officer_action"] == "REVIEW COMPLETED"


def test_add_officer_remarks(temp_db):
    """5. Test adding officer remarks without changing current review status."""
    create_review_record(bid_id="BID_REMARK_01", db_path=temp_db)
    # Start review first
    update_review_status(bid_id="BID_REMARK_01", action="REVIEW STARTED", db_path=temp_db)

    success, msg, rec = add_review_remark(
        bid_id="BID_REMARK_01",
        remarks="Discrepancy noted in turnover certificate; awaiting clarification.",
        db_path=temp_db
    )
    assert success is True
    assert rec["review_status"] == "UNDER REVIEW"
    assert "Discrepancy noted in turnover certificate" in rec["officer_remarks"]

    history = get_review_history("BID_REMARK_01", db_path=temp_db)
    actions = [h["action"] for h in history]
    assert "OFFICER REMARK ADDED" in actions


def test_audit_log_creation(temp_db):
    """6. Test that every action and creation event produces audit entries."""
    create_review_record(bid_id="BID_AUDIT_01", db_path=temp_db)
    update_review_status(bid_id="BID_AUDIT_01", action="REVIEW STARTED", remarks="Start", db_path=temp_db)
    update_review_status(bid_id="BID_AUDIT_01", action="INFORMATION REQUESTED", remarks="Need GST 3B", db_path=temp_db)

    history = get_review_history("BID_AUDIT_01", db_path=temp_db)
    # Initial creation + Review Started + Information Requested = 3 entries
    assert len(history) == 3
    assert history[0]["action"] == "RECORD CREATED"
    assert history[0]["previous_status"] == "NONE"
    assert history[0]["new_status"] == "NOT REVIEWED"

    assert history[1]["action"] == "REVIEW STARTED"
    assert history[1]["previous_status"] == "NOT REVIEWED"
    assert history[1]["new_status"] == "UNDER REVIEW"

    assert history[2]["action"] == "INFORMATION REQUESTED"
    assert history[2]["previous_status"] == "UNDER REVIEW"
    assert history[2]["new_status"] == "UNDER REVIEW"
    assert history[2]["remarks"] == "Need GST 3B"


def test_chronological_audit_history(temp_db):
    """7. Test that audit history is strictly chronological and timestamps are ordered."""
    create_review_record(bid_id="BID_CHRONO_01", db_path=temp_db)
    update_review_status(bid_id="BID_CHRONO_01", action="REVIEW STARTED", db_path=temp_db)
    update_review_status(bid_id="BID_CHRONO_01", action="ADDITIONAL VERIFICATION REQUIRED", db_path=temp_db)
    update_review_status(bid_id="BID_CHRONO_01", action="REVIEW COMPLETED", db_path=temp_db)

    history = get_review_history("BID_CHRONO_01", db_path=temp_db)
    assert len(history) == 4

    ids = [entry["id"] for entry in history]
    assert ids == sorted(ids), "Audit history entries must be strictly chronological by id"


def test_valid_status_transitions(temp_db):
    """8. Test comprehensive matrix of valid review transitions."""
    # NOT REVIEWED -> ESCALATED
    create_review_record(bid_id="BID_TRANS_01", db_path=temp_db)
    ok, msg, rec = update_review_status(
        bid_id="BID_TRANS_01",
        action="ESCALATED FOR FURTHER REVIEW",
        remarks="Blacklist flag triggered immediate escalation.",
        db_path=temp_db
    )
    assert ok is True
    assert rec["review_status"] == "ESCALATED"

    # ESCALATED -> REVIEW STARTED (resuming review)
    ok2, msg2, rec2 = update_review_status(
        bid_id="BID_TRANS_01",
        action="REVIEW STARTED",
        remarks="Officer resuming escalated review.",
        db_path=temp_db
    )
    assert ok2 is True
    assert rec2["review_status"] == "UNDER REVIEW"

    # UNDER REVIEW -> REVIEW COMPLETED
    ok3, msg3, rec3 = update_review_status(
        bid_id="BID_TRANS_01",
        action="REVIEW COMPLETED",
        remarks="Resolved satisfactorily.",
        db_path=temp_db
    )
    assert ok3 is True
    assert rec3["review_status"] == "REVIEWED"


def test_invalid_status_transitions(temp_db):
    """9. Test that invalid status transitions are rejected without crashing."""
    create_review_record(bid_id="BID_INVALID_01", db_path=temp_db)

    # Attempting REVIEW COMPLETED directly from NOT REVIEWED must fail validation
    ok, msg, rec = update_review_status(
        bid_id="BID_INVALID_01",
        action="REVIEW COMPLETED",
        remarks="Trying to skip review start.",
        db_path=temp_db
    )
    assert ok is False
    assert "Review action cannot be applied from the current status" in msg
    assert rec["review_status"] == "NOT REVIEWED"  # Status must remain unchanged

    # Invalid action
    is_valid, _, err = validate_review_action("NOT REVIEWED", "DISQUALIFY BIDDER")
    assert is_valid is False
    assert "Invalid officer action" in err


def test_review_statistics(temp_db):
    """10. Test calculating review metrics across multiple records."""
    create_review_record(bid_id="BID_STAT_1", db_path=temp_db) # NOT REVIEWED
    create_review_record(bid_id="BID_STAT_2", db_path=temp_db)
    update_review_status(bid_id="BID_STAT_2", action="REVIEW STARTED", db_path=temp_db) # UNDER REVIEW

    create_review_record(bid_id="BID_STAT_3", db_path=temp_db)
    update_review_status(bid_id="BID_STAT_3", action="REVIEW STARTED", db_path=temp_db)
    update_review_status(bid_id="BID_STAT_3", action="REVIEW COMPLETED", db_path=temp_db) # REVIEWED

    create_review_record(bid_id="BID_STAT_4", db_path=temp_db)
    update_review_status(bid_id="BID_STAT_4", action="ESCALATED FOR FURTHER REVIEW", db_path=temp_db) # ESCALATED

    stats = calculate_review_statistics(db_path=temp_db)
    assert stats["total_reviews"] == 4
    assert stats["not_reviewed"] == 1
    assert stats["under_review"] == 1
    assert stats["reviewed"] == 1
    assert stats["escalated"] == 1
    assert stats["completion_rate"] == 25.0  # 1/4 * 100


def test_empty_database_handling(temp_db):
    """11. Test that review functions handle an empty database gracefully without crashing."""
    stats = calculate_review_statistics(db_path=temp_db)
    assert stats["total_reviews"] == 0
    assert stats["not_reviewed"] == 0
    assert stats["under_review"] == 0
    assert stats["reviewed"] == 0
    assert stats["escalated"] == 0
    assert stats["completion_rate"] == 0.0

    history = get_review_history("BID_DOES_NOT_EXIST", db_path=temp_db)
    assert history == []

    all_reviews = list_all_reviews(db_path=temp_db)
    assert all_reviews == []


def test_disclaimer_present():
    """Verify that human-in-the-loop statutory disclaimer is preserved and non-decisional."""
    assert "AI provides decision support only" in WORKFLOW_DISCLAIMER
    assert "Procurement Officer" in WORKFLOW_DISCLAIMER
    assert "decision support" in WORKFLOW_DISCLAIMER.lower()
