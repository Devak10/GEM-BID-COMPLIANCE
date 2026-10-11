"""
AI-Powered Integrated Bid Compliance Verification Platform for GeM Procurement
AIML Hackathon Problem Statement ID: 26100
Organization: Chennai Petroleum Corporation Limited (CPCL)
Theme: Smart Automation

Module: Bid Review Workflow & Audit Trail
File: src/review_workflow.py

Provides human-in-the-loop review workflow and chronological audit management:
- Local SQLite persistence for bid review records and immutable audit logs
- Status transitions with strict workflow validation
- Audit history tracking every officer action and remark
- Non-decision-making workflow: AI provides decision support; authorized Procurement
  Officer retains full discretion for statutory decisions.
"""

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd


# Default database path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = str(PROJECT_ROOT / "data" / "review_audit.db")

# Supported review statuses
VALID_REVIEW_STATUSES = [
    "NOT REVIEWED",
    "UNDER REVIEW",
    "REVIEWED",
    "ESCALATED"
]

# Supported officer actions
VALID_OFFICER_ACTIONS = [
    "REVIEW STARTED",
    "INFORMATION REQUESTED",
    "ADDITIONAL VERIFICATION REQUIRED",
    "REVIEW COMPLETED",
    "ESCALATED FOR FURTHER REVIEW"
]

# Action to target status mapping
ACTION_STATUS_TARGETS = {
    "REVIEW STARTED": "UNDER REVIEW",
    "INFORMATION REQUESTED": "UNDER REVIEW",
    "ADDITIONAL VERIFICATION REQUIRED": "UNDER REVIEW",
    "REVIEW COMPLETED": "REVIEWED",
    "ESCALATED FOR FURTHER REVIEW": "ESCALATED"
}

# Human-in-the-loop statutory disclaimer
WORKFLOW_DISCLAIMER = (
    "AI provides decision support only. Final qualification or disqualification must be "
    "determined by the authorized Procurement Officer."
)


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Creates and returns a connection to the SQLite database.
    Ensures the parent directory exists.
    """
    path_to_use = db_path if db_path else DEFAULT_DB_PATH
    parent_dir = os.path.dirname(os.path.abspath(path_to_use))
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    
    conn = sqlite3.connect(path_to_use, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """
    Initializes database schema with bid_reviews and review_audit_log tables.
    Automatically runs on module access if tables do not exist.
    """
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.cursor()
            # Table 1: bid_reviews
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bid_reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bid_id TEXT UNIQUE NOT NULL,
                    tender_id TEXT,
                    company_name TEXT,
                    predicted_compliance TEXT,
                    compliance_probability REAL,
                    non_compliance_probability REAL,
                    risk_level TEXT,
                    review_priority TEXT,
                    review_status TEXT NOT NULL DEFAULT 'NOT REVIEWED',
                    officer_action TEXT NOT NULL DEFAULT 'PENDING REVIEW',
                    officer_remarks TEXT DEFAULT '',
                    document_completeness REAL,
                    documentation_consistency TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Table 2: review_audit_log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS review_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bid_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    previous_status TEXT NOT NULL,
                    new_status TEXT NOT NULL,
                    remarks TEXT DEFAULT '',
                    timestamp TEXT NOT NULL
                )
            """)

            # Indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_reviews_bid_id ON bid_reviews(bid_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_bid_id ON review_audit_log(bid_id)")
    finally:
        conn.close()


def validate_review_action(
    current_status: str,
    action: str,
    target_status: Optional[str] = None
) -> Tuple[bool, str, str]:
    """
    Validates whether an officer action and resulting status are valid from the current status.
    
    Returns:
        (is_valid, resolved_new_status, message)
    """
    norm_current = str(current_status or "").strip().upper()
    norm_action = str(action or "").strip().upper()
    
    if norm_current not in VALID_REVIEW_STATUSES:
        norm_current = "NOT REVIEWED"
    
    if norm_action not in VALID_OFFICER_ACTIONS:
        return False, norm_current, f"Invalid officer action '{action}'. Action must be one of {VALID_OFFICER_ACTIONS}."

    # Determine standard target status for this action
    derived_target = ACTION_STATUS_TARGETS.get(norm_action, norm_current)
    resolved_target = str(target_status).strip().upper() if target_status else derived_target

    if resolved_target not in VALID_REVIEW_STATUSES:
        return False, norm_current, f"Invalid target status '{target_status}'. Must be one of {VALID_REVIEW_STATUSES}."

    # Rule 1: From NOT REVIEWED
    if norm_current == "NOT REVIEWED":
        if norm_action == "REVIEW STARTED" and resolved_target in ["UNDER REVIEW", "NOT REVIEWED"]:
            return True, "UNDER REVIEW", "Review initiated successfully."
        elif norm_action == "ESCALATED FOR FURTHER REVIEW":
            return True, "ESCALATED", "Bid escalated directly for further review."
        else:
            return (
                False,
                norm_current,
                "Review action cannot be applied from the current status. Review must be started before completing or requesting information."
            )

    # Rule 2: From UNDER REVIEW
    elif norm_current == "UNDER REVIEW":
        if norm_action in ["INFORMATION REQUESTED", "ADDITIONAL VERIFICATION REQUIRED"]:
            return True, "UNDER REVIEW", f"Action '{norm_action}' recorded under ongoing review."
        elif norm_action == "REVIEW COMPLETED":
            return True, "REVIEWED", "Review completed by Procurement Officer."
        elif norm_action == "ESCALATED FOR FURTHER REVIEW":
            return True, "ESCALATED", "Bid escalated for supervisor/external review."
        elif norm_action == "REVIEW STARTED":
            return True, "UNDER REVIEW", "Review already in progress; status maintained."
        else:
            return False, norm_current, "Review action cannot be applied from the current status."

    # Rule 3: From REVIEWED
    elif norm_current == "REVIEWED":
        if norm_action == "REVIEW COMPLETED":
            return True, "REVIEWED", "Review status maintained with updated officer remarks."
        elif norm_action == "ESCALATED FOR FURTHER REVIEW":
            return True, "ESCALATED", "Previously reviewed bid escalated for higher review."
        elif norm_action in ["ADDITIONAL VERIFICATION REQUIRED", "INFORMATION REQUESTED"]:
            return True, "UNDER REVIEW", f"Bid reopened for {norm_action.lower()}."
        elif norm_action == "REVIEW STARTED":
            return False, norm_current, "Review action cannot be applied from the current status. Bid has already been reviewed."
        else:
            return False, norm_current, "Review action cannot be applied from the current status."

    # Rule 4: From ESCALATED
    elif norm_current == "ESCALATED":
        if norm_action == "ESCALATED FOR FURTHER REVIEW":
            return True, "ESCALATED", "Escalation notes updated."
        elif norm_action in ["REVIEW STARTED", "ADDITIONAL VERIFICATION REQUIRED", "INFORMATION REQUESTED"]:
            return True, "UNDER REVIEW", f"Escalated bid returned to active review via '{norm_action}'."
        elif norm_action == "REVIEW COMPLETED":
            return True, "REVIEWED", "Escalated review finalized and closed."
        else:
            return False, norm_current, "Review action cannot be applied from the current status."

    return False, norm_current, "Review action cannot be applied from the current status."


def create_review_record(
    bid_id: str,
    tender_id: str = "TENDER_UNKNOWN",
    company_name: str = "VENDOR_UNKNOWN",
    predicted_compliance: Union[str, int] = "Non-Compliant",
    compliance_probability: float = 0.5,
    non_compliance_probability: float = 0.5,
    risk_level: str = "Medium",
    review_priority: str = "PRIORITY REVIEW",
    officer_remarks: str = "",
    document_completeness: Optional[float] = None,
    documentation_consistency: Optional[str] = None,
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Creates a new review record in bid_reviews and logs the initial audit entry.
    If the bid record already exists, updates AI prediction metadata while preserving
    existing review status and audit trail.
    """
    init_db(db_path)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Format predicted compliance as text
    if isinstance(predicted_compliance, (int, float)):
        pred_label = "Compliant" if int(predicted_compliance) == 1 else "Non-Compliant"
    else:
        pred_label = str(predicted_compliance)

    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.cursor()
            # Check if record already exists
            cursor.execute("SELECT * FROM bid_reviews WHERE bid_id = ?", (bid_id,))
            existing = cursor.fetchone()

            if existing is not None:
                # Update AI signals and document info, preserve officer status and remarks
                cursor.execute("""
                    UPDATE bid_reviews
                    SET tender_id = ?,
                        company_name = ?,
                        predicted_compliance = ?,
                        compliance_probability = ?,
                        non_compliance_probability = ?,
                        risk_level = ?,
                        review_priority = ?,
                        document_completeness = COALESCE(?, document_completeness),
                        documentation_consistency = COALESCE(?, documentation_consistency),
                        updated_at = ?
                    WHERE bid_id = ?
                """, (
                    tender_id,
                    company_name,
                    pred_label,
                    float(compliance_probability),
                    float(non_compliance_probability),
                    str(risk_level),
                    str(review_priority),
                    document_completeness,
                    documentation_consistency,
                    now_str,
                    bid_id
                ))
            else:
                # Insert brand new review record
                cursor.execute("""
                    INSERT INTO bid_reviews (
                        bid_id, tender_id, company_name, predicted_compliance,
                        compliance_probability, non_compliance_probability,
                        risk_level, review_priority, review_status,
                        officer_action, officer_remarks, document_completeness,
                        documentation_consistency, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'NOT REVIEWED', 'PENDING REVIEW', ?, ?, ?, ?, ?)
                """, (
                    bid_id,
                    tender_id,
                    company_name,
                    pred_label,
                    float(compliance_probability),
                    float(non_compliance_probability),
                    str(risk_level),
                    str(review_priority),
                    officer_remarks or "",
                    document_completeness,
                    documentation_consistency,
                    now_str,
                    now_str
                ))

                # Initial audit log entry
                cursor.execute("""
                    INSERT INTO review_audit_log (
                        bid_id, action, previous_status, new_status, remarks, timestamp
                    ) VALUES (?, 'RECORD CREATED', 'NONE', 'NOT REVIEWED', 'Review record initialized from AI assessment.', ?)
                """, (bid_id, now_str))

        # Retrieve and return record
        return get_review_record(bid_id, db_path) or {}
    finally:
        conn.close()


def update_review_status(
    bid_id: str,
    action: str,
    remarks: str = "",
    target_status: Optional[str] = None,
    db_path: Optional[str] = None
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Updates review status for a bid with workflow validation and logs an audit entry.
    
    Returns:
        (success: bool, message: str, updated_record: Optional[Dict])
    """
    init_db(db_path)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    current_record = get_review_record(bid_id, db_path)
    if not current_record:
        return False, f"Review record for bid '{bid_id}' not found.", None

    current_status = current_record.get("review_status", "NOT REVIEWED")
    is_valid, new_status, validation_msg = validate_review_action(current_status, action, target_status)

    if not is_valid:
        return False, validation_msg, current_record

    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.cursor()
            # Update review record
            cursor.execute("""
                UPDATE bid_reviews
                SET review_status = ?,
                    officer_action = ?,
                    officer_remarks = ?,
                    updated_at = ?
                WHERE bid_id = ?
            """, (new_status, action, remarks, now_str, bid_id))

            # Insert audit log entry
            cursor.execute("""
                INSERT INTO review_audit_log (
                    bid_id, action, previous_status, new_status, remarks, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (bid_id, action, current_status, new_status, remarks or "", now_str))

        updated = get_review_record(bid_id, db_path)
        return True, validation_msg, updated
    finally:
        conn.close()


def add_review_remark(
    bid_id: str,
    remarks: str,
    db_path: Optional[str] = None
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Appends or updates officer remarks on a bid without altering review status,
    recording an audit log entry for transparency.
    """
    init_db(db_path)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    current_record = get_review_record(bid_id, db_path)
    if not current_record:
        return False, f"Review record for bid '{bid_id}' not found.", None

    current_status = current_record.get("review_status", "NOT REVIEWED")
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE bid_reviews
                SET officer_remarks = ?,
                    updated_at = ?
                WHERE bid_id = ?
            """, (remarks, now_str, bid_id))

            cursor.execute("""
                INSERT INTO review_audit_log (
                    bid_id, action, previous_status, new_status, remarks, timestamp
                ) VALUES (?, 'OFFICER REMARK ADDED', ?, ?, ?, ?)
            """, (bid_id, current_status, current_status, remarks, now_str))

        updated = get_review_record(bid_id, db_path)
        return True, "Officer remark recorded successfully.", updated
    finally:
        conn.close()


def get_review_record(bid_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieves a single bid review record by bid_id.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM bid_reviews WHERE bid_id = ?", (bid_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return dict(row)
    finally:
        conn.close()


def get_review_history(bid_id: str, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieves chronological audit trail for a given bid_id.
    Ordered by id / timestamp ascending.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, bid_id, action, previous_status, new_status, remarks, timestamp
            FROM review_audit_log
            WHERE bid_id = ?
            ORDER BY id ASC
        """, (bid_id,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def list_all_reviews(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieves all bid reviews ordered by last update descending.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM bid_reviews ORDER BY updated_at DESC")
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def calculate_review_statistics(db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Calculates aggregate workflow metrics across all review records:
    - total_reviews
    - not_reviewed
    - under_review
    - reviewed
    - escalated
    - completion_rate (percentage)
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                COUNT(*) as total_reviews,
                SUM(CASE WHEN review_status = 'NOT REVIEWED' THEN 1 ELSE 0 END) as not_reviewed,
                SUM(CASE WHEN review_status = 'UNDER REVIEW' THEN 1 ELSE 0 END) as under_review,
                SUM(CASE WHEN review_status = 'REVIEWED' THEN 1 ELSE 0 END) as reviewed,
                SUM(CASE WHEN review_status = 'ESCALATED' THEN 1 ELSE 0 END) as escalated
            FROM bid_reviews
        """)
        row = cursor.fetchone()
        if not row or row["total_reviews"] == 0:
            return {
                "total_reviews": 0,
                "not_reviewed": 0,
                "under_review": 0,
                "reviewed": 0,
                "escalated": 0,
                "completion_rate": 0.0
            }

        total = int(row["total_reviews"])
        reviewed = int(row["reviewed"] or 0)
        not_reviewed = int(row["not_reviewed"] or 0)
        under_review = int(row["under_review"] or 0)
        escalated = int(row["escalated"] or 0)

        rate = (reviewed / total * 100.0) if total > 0 else 0.0

        return {
            "total_reviews": total,
            "not_reviewed": not_reviewed,
            "under_review": under_review,
            "reviewed": reviewed,
            "escalated": escalated,
            "completion_rate": round(rate, 1)
        }
    finally:
        conn.close()
