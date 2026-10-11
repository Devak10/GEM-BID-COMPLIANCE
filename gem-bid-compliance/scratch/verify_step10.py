"""
Verification script for STEP 10: Hackathon-Ready UI/UX & System Polish.
Tests Streamlit app across all 7 views, tests all 6 archetypal demo profiles,
verifies model and dataset file preservation, and checks for zero SIH occurrences.
"""

import sys
import os
from pathlib import Path

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from streamlit.testing.v1 import AppTest
from src.predict_compliance import load_model, assess_bid
from src.explainability import explain_prediction
from src.review_workflow import (
    init_db,
    create_review_record,
    get_review_record,
    update_review_status,
    get_review_history,
    calculate_review_statistics,
    DEFAULT_DB_PATH
)

def test_step10_full_verification():
    print("==================================================")
    print("STEP 10 VERIFICATION: HACKATHON-READY UI/UX & SYSTEM POLISH")
    print("==================================================")

    # 1. Verify Model File Preservation
    model_path = PROJECT_ROOT / "models" / "compliance_model.pkl"
    assert model_path.exists(), "Model file not found!"
    model_size = model_path.stat().st_size
    print(f"[CHECK] Model file size: {model_size:,} bytes")
    assert model_size == 1259122, f"Model size mismatch! Expected 1,259,122 bytes, got {model_size}"

    # 2. Verify Dataset File Preservation
    dataset_path = PROJECT_ROOT / "data" / "bid_compliance_dataset.csv"
    assert dataset_path.exists(), "Dataset file not found!"
    dataset_size = dataset_path.stat().st_size
    print(f"[CHECK] Dataset file size: {dataset_size:,} bytes")
    assert dataset_size == 308841, f"Dataset size mismatch! Expected 308,841 bytes, got {dataset_size}"

    # 3. Verify SQLite DB path
    db_path = PROJECT_ROOT / "data" / "review_audit.db"
    print(f"[CHECK] Review DB path: {db_path} (exists={db_path.exists()})")

    # 4. Streamlit App Rendering via AppTest
    app_path = str(PROJECT_ROOT / "app" / "app.py")
    print("\n--- Testing Streamlit Pages Rendering ---")

    at = AppTest.from_file(app_path, default_timeout=15)
    at.run(timeout=15)
    assert not at.exception, f"Dashboard view error: {at.exception}"
    print("[PASS] Page 1: Dashboard rendered successfully with zero exceptions.")

    # Page 2: Bid Verification
    at.radio(key="nav_selection").set_value("Bid Verification")
    at.run(timeout=15)
    assert not at.exception, f"Bid Verification view error: {at.exception}"
    print("[PASS] Page 2: Bid Verification rendered successfully with zero exceptions.")

    # Page 3: Document Verification
    at.radio(key="nav_selection").set_value("Document Verification")
    at.run(timeout=15)
    assert not at.exception, f"Document Verification view error: {at.exception}"
    print("[PASS] Page 3: Document Verification rendered successfully with zero exceptions.")

    # Page 4: Risk Analysis
    at.radio(key="nav_selection").set_value("Risk Analysis")
    at.run(timeout=15)
    assert not at.exception, f"Risk Analysis view error: {at.exception}"
    print("[PASS] Page 4: Risk Analysis rendered successfully with zero exceptions.")

    # Page 5: Model Insights
    at.radio(key="nav_selection").set_value("Model Insights")
    at.run(timeout=15)
    assert not at.exception, f"Model Insights view error: {at.exception}"
    print("[PASS] Page 5: Model Insights rendered successfully with zero exceptions.")

    # Page 6: Officer Review
    at.radio(key="nav_selection").set_value("Officer Review")
    at.run(timeout=15)
    assert not at.exception, f"Officer Review view error: {at.exception}"
    print("[PASS] Page 6: Officer Review rendered successfully with zero exceptions.")

    # Page 7: About System
    at.radio(key="nav_selection").set_value("About System")
    at.run(timeout=15)
    assert not at.exception, f"About System view error: {at.exception}"
    print("[PASS] Page 7: About System rendered successfully with zero exceptions.")

    # 5. Test 6 Archetypal Demo Profiles
    print("\n--- Testing 6 Archetypal Demo Profiles In Model & Rule Assessment ---")
    model = load_model(str(model_path))
    df = pd.read_csv(dataset_path)

    archetypes = [
        ("Compliant Bidder", "BID0001"),
        ("Incomplete Documentation", "BID0008"),
        ("High-Risk Bidder", "BID0007"),
        ("Blacklisted Vendor", "BID0073"),
        ("Debarred Vendor", "BID0033"),
        ("Ineligible Bidder", "BID0021"),
    ]

    results_table = []
    for label, b_id in archetypes:
        row = df[df["bid_id"] == b_id].iloc[0].to_dict()
        assessment = assess_bid(row, model=model)
        explanation = explain_prediction(row, model=model)
        pred_raw = assessment["prediction"]["predicted_compliance"]
        pred = "Compliant" if pred_raw == 1 else "Non-Compliant"
        risk = assessment["risk_assessment"]["risk_level"]
        prio = explanation["review_priority"]
        if not isinstance(prio, str):
            prio = prio.get("priority_label", "STANDARD REVIEW")
        prob = assessment["prediction"]["compliance_probability"] * 100
        
        print(f"  [{label}] Bid ID: {b_id} | Co: {row['company_name'][:25]} | Pred: {pred} ({pred_raw}) | Prob: {prob:.1f}% | Risk: {risk} | Priority: {prio}")
        results_table.append({
            "Profile": label,
            "Bid ID": b_id,
            "Company": row["company_name"],
            "Prediction": pred,
            "Probability": f"{prob:.1f}%",
            "Risk Level": risk,
            "Priority": prio
        })

    assert len(results_table) == 6, "Expected 6 evaluated profiles"

    # 6. Test Review Workflow Operations
    print("\n--- Testing Review Workflow Operations & State Machine ---")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        temp_db_path = tmp_db.name

    try:
        init_db(temp_db_path)
        test_bid_id = "BID_STEP10_VERIFY"
        rec = create_review_record(
            bid_id=test_bid_id,
            tender_id="GEM/2026/B/8888",
            company_name="Step 10 Audit Verification Co",
            predicted_compliance="Non-Compliant",
            compliance_probability=0.25,
            non_compliance_probability=0.75,
            risk_level="High",
            review_priority="HIGH PRIORITY REVIEW",
            db_path=temp_db_path
        )
        assert rec["review_status"] == "NOT REVIEWED"
        print("  [OK] Initial record created as NOT REVIEWED")

        ok, msg, updated = update_review_status(test_bid_id, "REVIEW STARTED", remarks="Starting examination of GST and OEM proof", target_status="UNDER REVIEW", db_path=temp_db_path)
        assert ok and updated["review_status"] == "UNDER REVIEW"
        print("  [OK] Transitioned to UNDER REVIEW via 'REVIEW STARTED'")

        ok, msg, updated = update_review_status(test_bid_id, "ESCALATED FOR FURTHER REVIEW", remarks="Turnover discrepancy flagged for committee review", target_status="ESCALATED", db_path=temp_db_path)
        assert ok and updated["review_status"] == "ESCALATED"
        print("  [OK] Transitioned to ESCALATED via 'ESCALATED FOR FURTHER REVIEW'")

        hist = get_review_history(test_bid_id, db_path=temp_db_path)
        assert len(hist) == 3, f"Expected 3 history items, got {len(hist)}"
        print(f"  [OK] Immutable audit log contains {len(hist)} chronological events")
    finally:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)

    # 7. Check Prohibited Terms Search
    print("\n--- Checking for Prohibited Term 'SIH' Across Codebase ---")
    sih_found = False
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Skip git, venv, cache, scratch
        if any(skip in root for skip in [".git", "venv", ".pytest_cache", "__pycache__", "scratch"]):
            continue
        for f in files:
            if f.endswith((".py", ".md", ".txt", ".json", ".csv")):
                f_path = Path(root) / f
                try:
                    content = f_path.read_text(encoding="utf-8", errors="ignore")
                    if "SIH" in content.upper() and "SIH" in content:
                        print(f"  [WARNING] Found 'SIH' in {f_path}")
                        sih_found = True
                except Exception:
                    pass

    assert not sih_found, "Prohibited term 'SIH' found in codebase!"
    print("[PASS] ZERO occurrences of 'SIH' found in workspace.")

    print("\n==================================================")
    print("ALL STEP 10 VERIFICATION CHECKS PASSED WITH ZERO ERRORS!")
    print("==================================================")

if __name__ == "__main__":
    test_step10_full_verification()
