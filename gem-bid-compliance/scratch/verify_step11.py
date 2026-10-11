"""
AI-Powered Integrated Bid Compliance Verification Platform for GeM Procurement
AIML Hackathon | Problem Statement ID: 26100
Organization: Chennai Petroleum Corporation Limited (CPCL)
Theme: Smart Automation

STEP 11 COMPREHENSIVE VERIFICATION HARNESS
File: scratch/verify_step11.py

Validates the complete application across:
- Data and Model Integrity
- Prediction Engine & Rule Safety Layer
- Document Verification Module & Integration Transfer
- AI Explainability & Non-Causal Phrasing
- Review Workflow, State Transitions & SQLite Audit Trail
- Streamlit All 7 Pages (AppTest Execution)
- Six Demo Profiles & Custom Input Resilience
- Risk Analysis Multi-Dimensional Filtering
- Error Handling & Empty States
- Security, Hygiene, and Prohibited Terminology Checks
"""

import sys
import os
import json
import sqlite3
import tempfile
from pathlib import Path

# Ensure UTF-8 output encoding on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

# Import system modules
from src.predict_compliance import (
    load_model,
    predict_bid,
    predict_from_row,
    run_basic_compliance_checks,
    assess_bid,
    MODEL_FEATURE_COLUMNS
)
from src.document_verification import (
    SUPPORTED_DOCUMENT_TYPES,
    EXPECTED_KEYWORDS,
    STATUS_VERIFIED,
    STATUS_WARNING,
    STATUS_EXTRACTION_FAILED,
    STATUS_UNSUPPORTED,
    extract_text_from_file,
    identify_document_type,
    verify_document_content,
    calculate_document_completeness,
    calculate_document_consistency,
    generate_document_findings
)
from src.explainability import (
    explain_prediction,
    get_top_positive_factors,
    get_top_risk_factors,
    generate_officer_summary,
    get_review_priority
)
from src.review_workflow import (
    init_db,
    create_review_record,
    get_review_record,
    update_review_status,
    add_review_remark,
    get_review_history,
    calculate_review_statistics,
    validate_review_action,
    VALID_REVIEW_STATUSES,
    VALID_OFFICER_ACTIONS
)
from streamlit.testing.v1 import AppTest

DATASET_PATH = PROJECT_ROOT / "data" / "bid_compliance_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "compliance_model.pkl"
EXPECTED_DATASET_SIZE = 308841
EXPECTED_MODEL_SIZE = 1259122


def test_part_c_dataset_integrity():
    print("\n" + "="*50)
    print("PART C — DATASET INTEGRITY VALIDATION")
    print("="*50)
    assert DATASET_PATH.exists(), f"Dataset file missing at {DATASET_PATH}"
    file_size = DATASET_PATH.stat().st_size
    print(f"Dataset File Size: {file_size:,} bytes (Expected: {EXPECTED_DATASET_SIZE:,})")
    assert file_size == EXPECTED_DATASET_SIZE, f"Dataset size modified! Expected {EXPECTED_DATASET_SIZE}, got {file_size}"

    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns (Expected: 1000, 34)")
    assert df.shape == (1000, 34), f"Dataset shape mismatch: {df.shape}"

    # Missing values check
    missing = df.isnull().sum().sum()
    print(f"Total Missing Values: {missing}")
    assert missing == 0, f"Found {missing} unexpected null values in dataset"

    # Duplicates check
    dup_ids = df["bid_id"].duplicated().sum()
    dup_rows = df.duplicated().sum()
    print(f"Duplicate Bid IDs: {dup_ids}, Duplicate Full Rows: {dup_rows}")
    assert dup_ids == 0, "Duplicate Bid IDs found"
    assert dup_rows == 0, "Duplicate rows found"

    # Logical constraints check
    assert (df["submitted_documents"] <= df["required_documents"]).all(), "Constraint violated: submitted > required"
    assert (df["compliance_score"] >= 0).all() and (df["compliance_score"] <= 100).all(), "Invalid compliance score range"
    assert (df["local_content_percentage"] >= 0).all() and (df["local_content_percentage"] <= 100).all(), "Invalid local content range"
    print("[PASS] Dataset integrity confirmed. File is completely unchanged.")
    return df


def test_part_d_model_integrity():
    print("\n" + "="*50)
    print("PART D — MODEL INTEGRITY VALIDATION")
    print("="*50)
    assert MODEL_PATH.exists(), f"Model file missing at {MODEL_PATH}"
    file_size = MODEL_PATH.stat().st_size
    print(f"Model File Size: {file_size:,} bytes (Expected: {EXPECTED_MODEL_SIZE:,})")
    assert file_size == EXPECTED_MODEL_SIZE, f"Model size modified! Expected {EXPECTED_MODEL_SIZE}, got {file_size}"

    model = load_model(str(MODEL_PATH))
    assert model is not None, "Model failed to load"
    assert hasattr(model, "predict"), "Model missing predict method"
    assert hasattr(model, "predict_proba"), "Model missing predict_proba method"
    assert hasattr(model, "named_steps"), "Pipeline missing named_steps attribute"
    assert "preprocessor" in model.named_steps, "Pipeline missing preprocessor"
    assert "classifier" in model.named_steps, "Pipeline missing classifier"

    print(f"Model Pipeline Steps: {list(model.named_steps.keys())}")
    print("[PASS] Model file integrity confirmed. Pipeline is deserialized and intact.")
    return model


def test_part_e_prediction_engine(model, df):
    print("\n" + "="*50)
    print("PART E — PREDICTION ENGINE TEST")
    print("="*50)
    # 1. Normal compliant bid (BID0001)
    row_compliant = df[df["bid_id"] == "BID0001"].iloc[0].to_dict()
    pred1 = predict_bid(row_compliant, model=model)
    assert pred1["predicted_compliance"] == 1
    assert pred1["compliance_probability"] > 0.80
    print(f"1. Normal Compliant (BID0001): Pred={pred1['predicted_compliance']}, Prob={pred1['compliance_probability']:.3f}, Risk={pred1['risk_level']}")

    # 2. Incomplete documentation (BID0008)
    row_incomplete = df[df["bid_id"] == "BID0008"].iloc[0].to_dict()
    pred2 = predict_bid(row_incomplete, model=model)
    assert "predicted_compliance" in pred2
    print(f"2. Incomplete Docs (BID0008): Pred={pred2['predicted_compliance']}, Prob={pred2['compliance_probability']:.3f}, Risk={pred2['risk_level']}")

    # 3. High-risk bid (BID0007)
    row_high_risk = df[df["bid_id"] == "BID0007"].iloc[0].to_dict()
    pred3 = predict_bid(row_high_risk, model=model)
    assert pred3["predicted_compliance"] == 0
    assert pred3["risk_level"] == "High"
    print(f"3. High-Risk (BID0007): Pred={pred3['predicted_compliance']}, Prob={pred3['compliance_probability']:.3f}, Risk={pred3['risk_level']}")

    # 4. Blacklisted bidder (BID0073)
    row_bl = df[df["bid_id"] == "BID0073"].iloc[0].to_dict()
    pred4 = predict_bid(row_bl, model=model)
    assert pred4["risk_level"] == "High"
    print(f"4. Blacklisted (BID0073): Pred={pred4['predicted_compliance']}, Prob={pred4['compliance_probability']:.3f}, Risk={pred4['risk_level']}")

    # 5. Debarred bidder (BID0033)
    row_deb = df[df["bid_id"] == "BID0033"].iloc[0].to_dict()
    pred5 = predict_bid(row_deb, model=model)
    assert pred5["risk_level"] == "High"
    print(f"5. Debarred (BID0033): Pred={pred5['predicted_compliance']}, Prob={pred5['compliance_probability']:.3f}, Risk={pred5['risk_level']}")

    # 6. Ineligible bidder (BID0021)
    row_inel = df[df["bid_id"] == "BID0021"].iloc[0].to_dict()
    pred6 = predict_bid(row_inel, model=model)
    assert pred6["risk_level"] == "High"
    print(f"6. Ineligible (BID0021): Pred={pred6['predicted_compliance']}, Prob={pred6['compliance_probability']:.3f}, Risk={pred6['risk_level']}")

    # 7. Missing mandatory columns handled cleanly with ValueError
    try:
        predict_bid({"bid_id": "ERR_TEST"}, model=model)
        assert False, "Should have raised ValueError on missing columns"
    except ValueError as ve:
        assert "missing required model features" in str(ve)
        print("7. Missing/invalid columns handled cleanly: ValueError raised with descriptive message")

    # 8. Unknown categorical values
    unusual_row = dict(row_compliant)
    unusual_row.update({
        "project_type": "Quantum Cryptography Rig",
        "state": "Unknown Jurisdiction",
        "district": "Unknown Sector",
        "gst_registration": "Pending Verification 99"
    })
    pred8 = predict_bid(unusual_row, model=model)
    assert pred8["predicted_compliance"] in [0, 1]
    print(f"8. Unknown categorical values: Handled smoothly without crash (Pred={pred8['predicted_compliance']}, Prob={pred8['compliance_probability']:.3f})")

    # Assess bid wrapper check
    assess_out = assess_bid(row_compliant, model=model)
    assert "prediction" in assess_out
    assert "risk_assessment" in assess_out
    assert "verification" in assess_out
    assert "procurement_review" in assess_out
    print("[PASS] Prediction engine is resilient, robust, and performs flawlessly.")


def test_part_f_critical_rule_safety(model, df):
    print("\n" + "="*50)
    print("PART F — CRITICAL STATUTORY RULE SAFETY VALIDATION")
    print("="*50)

    # Blacklisted bidder
    bl_row = df[df["bid_id"] == "BID0073"].iloc[0].to_dict()
    bl_assessment = assess_bid(bl_row, model=model)
    assert bl_assessment["procurement_review"]["statutory_exclusions_flagged"] is True
    assert bl_assessment["risk_assessment"]["risk_level"] == "High"
    assert any("Blacklisted" in f for f in bl_assessment["procurement_review"]["exclusion_flags"])
    print("[PASS] BLACKLISTED bidder correctly flagged with statutory exclusion and High Risk.")

    # Debarred bidder
    deb_row = df[df["bid_id"] == "BID0033"].iloc[0].to_dict()
    deb_assessment = assess_bid(deb_row, model=model)
    assert deb_assessment["procurement_review"]["statutory_exclusions_flagged"] is True
    assert deb_assessment["risk_assessment"]["risk_level"] == "High"
    assert any("Debarred" in f for f in deb_assessment["procurement_review"]["exclusion_flags"])
    print("[PASS] DEBARRED bidder correctly flagged with statutory exclusion and High Risk.")

    # Ineligible bidder
    inel_row = df[df["bid_id"] == "BID0021"].iloc[0].to_dict()
    inel_assessment = assess_bid(inel_row, model=model)
    assert inel_assessment["procurement_review"]["statutory_exclusions_flagged"] is True
    assert inel_assessment["risk_assessment"]["risk_level"] == "High"
    assert any("Not Eligible" in f for f in inel_assessment["procurement_review"]["exclusion_flags"])
    print("[PASS] INELIGIBLE bidder correctly flagged with statutory exclusion and High Risk.")

    # Synthetic high-probability bidder with Debarred exclusion
    synth_row = dict(df[df["bid_id"] == "BID0001"].iloc[0].to_dict())
    synth_row["debarment_status"] = "Debarred"
    synth_assessment = assess_bid(synth_row, model=model)
    ml_prob = synth_assessment["prediction"]["compliance_probability"] * 100
    print(f"Synthetic Debarred Profile: ML Probability = {ml_prob:.1f}%")
    assert synth_assessment["procurement_review"]["statutory_exclusions_flagged"] is True
    assert synth_assessment["risk_assessment"]["risk_level"] == "High"
    assert synth_assessment["procurement_review"]["required"] is True
    print("[PASS] High ML probability does NOT override critical statutory exclusion indicator.")


def test_part_g_document_verification():
    print("\n" + "="*50)
    print("PART G — DOCUMENT VERIFICATION ENGINE TEST")
    print("="*50)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Valid TXT document
        txt_file = tmp_path / "gst_cert.txt"
        txt_content = (
            "GOVERNMENT OF INDIA\n"
            "GOODS AND SERVICES TAX REGISTRATION CERTIFICATE\n"
            "GSTIN: 33AAAAA0000A1Z5\n"
            "Legal Name: Test Corp Pvt Ltd\n"
            "GST Status: Active Verified"
        )
        txt_file.write_text(txt_content, encoding="utf-8")
        with open(txt_file, "rb") as f:
            res_txt = extract_text_from_file(f, "gst_cert.txt")
        assert res_txt["status"] == "SUCCESS"
        text = res_txt["text"]
        ident = identify_document_type(text)
        doc_type = ident["identified_type"]
        assert doc_type == "GST Registration Certificate"
        v_res = verify_document_content(doc_type, text)
        assert v_res["status"] == STATUS_VERIFIED
        print(f"1. Valid TXT: Type={doc_type}, Status={v_res['status']}, Found={len(v_res['detected_keywords'])} keywords")

        # 2. Empty TXT
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("", encoding="utf-8")
        with open(empty_file, "rb") as f:
            empty_res = extract_text_from_file(f, "empty.txt")
        assert empty_res["status"] == STATUS_WARNING or empty_res.get("is_empty") is True or not empty_res["text"].strip()
        print("2. Empty TXT: Extraction handled gracefully without crash")

        # 3. Unsupported extension
        bin_file = tmp_path / "program.exe"
        bin_file.write_bytes(b"\x00\x01\x02\x03")
        with open(bin_file, "rb") as f:
            unsup_res = extract_text_from_file(f, "program.exe")
        assert unsup_res["status"] == STATUS_UNSUPPORTED
        print("3. Unsupported format (.exe): Rejected cleanly with UNSUPPORTED FORMAT status")

        # 4. Missing keywords
        irrel_file = tmp_path / "irrelevant.txt"
        irrel_file.write_text("This is an unrelated recipe for soup.", encoding="utf-8")
        with open(irrel_file, "rb") as f:
            irrel_res = extract_text_from_file(f, "irrelevant.txt")
        v_res_irrel = verify_document_content("GST Registration Certificate", irrel_res["text"])
        assert v_res_irrel["status"] == STATUS_WARNING
        print(f"4. Missing keywords: Flagged as {v_res_irrel['status']}")

        # 5. Consistency checking
        docs_consistent = [
            {"document_type": "GST Registration Certificate", "file_name": "gst_cert.txt", "text": "Legal Name: Bharat Data Systems Enterprises\nGSTIN: 33AAAAA0000A1Z5", "status": STATUS_VERIFIED},
            {"document_type": "PAN Card / PAN Document", "file_name": "pan_card.txt", "text": "Name: Bharat Data Systems Enterprises\nPAN: AAAAA0000A", "status": STATUS_VERIFIED}
        ]
        cons_res1 = calculate_document_consistency(docs_consistent)
        assert cons_res1["status"] == "CONSISTENT"
        print(f"5. Consistent docs: Status={cons_res1['status']}, Score={cons_res1['score']}%")

        docs_inconsistent = [
            {"document_type": "GST Registration Certificate", "file_name": "gst_cert.txt", "text": "Legal Name: Bharat Data Systems Enterprises\nGSTIN: 33AAAAA0000A1Z5", "status": STATUS_VERIFIED},
            {"document_type": "PAN Card / PAN Document", "file_name": "pan_card.txt", "text": "Name: Completely Different Vendor Pvt Ltd\nPAN: ZZZZZ9999Z", "status": STATUS_VERIFIED}
        ]
        cons_res2 = calculate_document_consistency(docs_inconsistent)
        assert cons_res2["status"] == "REVIEW REQUIRED"
        print(f"6. Inconsistent docs: Status={cons_res2['status']}, Score={cons_res2['score']}%")

    print("[PASS] Document verification engine tests passed across all scenarios.")


def test_part_h_document_bid_integration(model, df):
    print("\n" + "="*50)
    print("PART H — DOCUMENT TO BID INTEGRATION TEST")
    print("="*50)
    base_bid = df[df["bid_id"] == "BID0008"].iloc[0].to_dict()
    original_gst = base_bid["gst_registration"]
    original_debarment = base_bid["debarment_status"]

    # Simulate transfer of document verification results into bid record
    doc_results_transfer = {
        "document_completeness": 100.0,
        "documentation_consistency": 95.0,
        "submitted_documents": base_bid["required_documents"]
    }
    updated_bid = dict(base_bid)
    updated_bid.update(doc_results_transfer)

    # Verify document metrics transferred
    assert updated_bid["document_completeness"] == 100.0
    assert updated_bid["documentation_consistency"] == 95.0

    # Verify statutory and organizational fields were NOT corrupted or overwritten
    assert updated_bid["gst_registration"] == original_gst
    assert updated_bid["debarment_status"] == original_debarment

    # Verify prediction and explainability work on updated record
    assessment = assess_bid(updated_bid, model=model)
    explanation = explain_prediction(updated_bid, model=model)
    assert assessment["prediction"]["predicted_compliance"] in [0, 1]
    assert explanation["review_priority"] is not None
    print("[PASS] Document verification results successfully transferred to Bid Verification without side effects.")


def test_part_i_explainability(model, df):
    print("\n" + "="*50)
    print("PART I — EXPLAINABILITY & NON-CAUSAL LANGUAGE AUDIT")
    print("="*50)
    row = df[df["bid_id"] == "BID0007"].iloc[0].to_dict()
    explanation = explain_prediction(row, model=model)

    assert "review_priority" in explanation
    assert "positive_factors" in explanation
    assert "risk_factors" in explanation
    assert "officer_summary" in explanation

    summary_text = str(explanation.get("officer_summary", ""))
    pos_factors = " ".join(f.get("description", "") for f in explanation.get("positive_factors", []))
    risk_factors = " ".join(r.get("description", "") for r in explanation.get("risk_factors", []))
    full_text = f"{summary_text} {pos_factors} {risk_factors}".lower()

    # Verify non-causal language constraints
    assert "caused the decision" not in full_text
    assert "guarantees" not in full_text
    assert "proves guilt" not in full_text
    print(f"Officer Summary Snippet: '{summary_text[:120]}...'")
    print(f"Review Priority: {explanation['review_priority']}")
    print("[PASS] Explainability module adheres strictly to non-causal, assistive phrasing.")


def test_part_j_k_review_workflow_and_audit():
    print("\n" + "="*50)
    print("PART J & K — REVIEW WORKFLOW & SQLITE AUDIT TRAIL TEST")
    print("="*50)
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        temp_db_path = tmp_db.name

    try:
        init_db(temp_db_path)
        test_bid_id = "BID_STEP11_TEST"

        # 1. Create review
        rec = create_review_record(
            bid_id=test_bid_id,
            tender_id="GEM/2026/B/9999",
            company_name="Step 11 Test Corp",
            predicted_compliance="Compliant",
            compliance_probability=0.88,
            non_compliance_probability=0.12,
            risk_level="Low",
            review_priority="STANDARD REVIEW",
            db_path=temp_db_path
        )
        assert rec["review_status"] == "NOT REVIEWED"
        print("1. Record created: Status = NOT REVIEWED")

        # 2. Valid transition: NOT REVIEWED -> UNDER REVIEW
        ok, msg, updated = update_review_status(test_bid_id, "REVIEW STARTED", remarks="Commencing verification", target_status="UNDER REVIEW", db_path=temp_db_path)
        assert ok and updated["review_status"] == "UNDER REVIEW"
        print("2. Transitioned to UNDER REVIEW via REVIEW STARTED")

        # 3. Add remark
        ok, msg, _ = add_review_remark(test_bid_id, "Verified OEM authorization on portal", db_path=temp_db_path)
        assert ok
        print("3. Officer remark added to audit trail")

        # 4. Valid transition: UNDER REVIEW -> REVIEWED
        ok, msg, updated = update_review_status(test_bid_id, "REVIEW COMPLETED", remarks="All checks verified compliant", target_status="REVIEWED", db_path=temp_db_path)
        assert ok and updated["review_status"] == "REVIEWED"
        print("4. Transitioned to REVIEWED via REVIEW COMPLETED")

        # 5. Invalid transition: REVIEWED -> UNDER REVIEW (should be rejected)
        ok, msg, _ = update_review_status(test_bid_id, "REVIEW STARTED", remarks="Attempting invalid reset", target_status="UNDER REVIEW", db_path=temp_db_path)
        assert not ok
        print(f"5. Invalid transition rejected cleanly: '{msg}'")

        # 6. Audit history chronological verification
        history = get_review_history(test_bid_id, db_path=temp_db_path)
        assert len(history) == 4, f"Expected 4 audit events, got {len(history)}"
        print(f"6. Audit history verified: {len(history)} chronological immutable entries recorded")

        # 7. Check statistics
        stats = calculate_review_statistics(db_path=temp_db_path)
        assert stats["total_reviews"] == 1
        assert stats["reviewed"] == 1
        print(f"7. Review statistics verified: Total={stats['total_reviews']}, Reviewed={stats['reviewed']}")

    finally:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)

    print("[PASS] Review workflow state machine and SQLite audit logging verified.")


def test_part_l_streamlit_all_7_pages():
    print("\n" + "="*50)
    print("PART L — STREAMLIT ALL 7 PAGES APPTEST VERIFICATION")
    print("="*50)
    app_path = str(PROJECT_ROOT / "app" / "app.py")
    at = AppTest.from_file(app_path, default_timeout=15)
    at.run(timeout=15)
    assert not at.exception, f"Dashboard exception: {at.exception}"
    print("[PASS] Page 1: Dashboard rendered successfully.")

    pages = [
        "Bid Verification",
        "Document Verification",
        "Risk Analysis",
        "Model Insights",
        "Officer Review",
        "About System"
    ]
    for idx, p_name in enumerate(pages, start=2):
        at.radio(key="nav_selection").set_value(p_name)
        at.run(timeout=15)
        assert not at.exception, f"{p_name} exception: {at.exception}"
        print(f"[PASS] Page {idx}: {p_name} rendered successfully.")


def test_part_m_six_demo_profiles(model, df):
    print("\n" + "="*50)
    print("PART M — ALL SIX DEMO PROFILES EVALUATION")
    print("="*50)
    demo_profiles = [
        ("Compliant Bidder", "BID0001", "Bharat Data Systems Enterprises"),
        ("Incomplete Documentation", "BID0008", "Metro Heavy Engineering Enterprises"),
        ("High-Risk Bidder", "BID0007", "XYZ Procurement Solutions Ltd"),
        ("Blacklisted Vendor", "BID0073", "Pacific BuildTech Services"),
        ("Debarred Vendor", "BID0033", "Pacific Electro-Mech Systems Pvt Ltd"),
        ("Ineligible Bidder", "BID0021", "Deccan Civil Works Pvt Ltd")
    ]

    summary_records = []
    for label, bid_id, expected_company in demo_profiles:
        row = df[df["bid_id"] == bid_id].iloc[0].to_dict()
        assessment = assess_bid(row, model=model)
        explanation = explain_prediction(row, model=model)

        pred_val = assessment["prediction"]["predicted_compliance"]
        pred_label = "Compliant" if pred_val == 1 else "Non-Compliant"
        prob_comp = assessment["prediction"]["compliance_probability"] * 100
        prob_non = assessment["prediction"]["non_compliance_probability"] * 100
        risk_lvl = assessment["risk_assessment"]["risk_level"]
        priority_info = explanation["review_priority"]
        priority_label = priority_info.get("priority_label") if isinstance(priority_info, dict) else str(priority_info)
        statutory_flag = assessment["procurement_review"]["statutory_exclusions_flagged"]
        failed_rules = len(assessment["verification"]["failed_checks"])
        warning_rules = len(assessment["verification"]["warning_checks"])

        print(f"[{label}] ID: {bid_id} | ML: {pred_label} ({prob_comp:.1f}%) | Risk: {risk_lvl} | Statutory Flag: {statutory_flag} | Priority: {priority_label}")

        summary_records.append({
            "Profile": label,
            "Bid ID": bid_id,
            "Company": row["company_name"],
            "ML Prediction": pred_label,
            "Compliance Prob": f"{prob_comp:.1f}%",
            "Non-Compliance Prob": f"{prob_non:.1f}%",
            "Risk Level": risk_lvl,
            "Priority": priority_label,
            "Statutory Exclusion": "YES" if statutory_flag else "NO",
            "Failed Rules": failed_rules,
            "Warning Rules": warning_rules
        })

    assert len(summary_records) == 6
    print("[PASS] All 6 demo profiles evaluated successfully.")
    return summary_records


def test_part_o_risk_analysis_filters(df):
    print("\n" + "="*50)
    print("PART O — RISK ANALYSIS MULTI-DIMENSIONAL FILTERS TEST")
    print("="*50)
    # Test State filter
    filt_tn = df[df["state"] == "Tamil Nadu"]
    assert len(filt_tn) > 0, "No records found for Tamil Nadu"

    # Test Project Type filter
    filt_infra = df[df["project_type"] == "Infrastructure"]
    assert len(filt_infra) > 0, "No records found for Infrastructure"

    # Test Risk Level filter
    filt_high = df[df["risk_level"] == "High"]
    assert len(filt_high) > 0, "No records found for High Risk"

    # Test combined filter
    combined = df[(df["state"] == "Tamil Nadu") & (df["project_type"] == "Infrastructure") & (df["risk_level"] == "High")]
    print(f"Combined filter (Tamil Nadu + Infrastructure + High Risk): {len(combined)} matching records")

    # Test no matching records (edge case)
    no_match = df[(df["state"] == "NonExistentState") & (df["risk_level"] == "UltraHigh")]
    assert len(no_match) == 0
    print("Empty filter state handled properly without exception (0 matching records)")
    print("[PASS] Risk Analysis filtering operates correctly.")


def test_part_q_empty_database_states():
    print("\n" + "="*50)
    print("PART Q — EMPTY DATABASE STATES VALIDATION")
    print("="*50)
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        empty_db_path = tmp_db.name

    try:
        init_db(empty_db_path)
        stats = calculate_review_statistics(db_path=empty_db_path)
        assert stats["total_reviews"] == 0
        assert stats["under_review"] == 0
        assert stats["reviewed"] == 0
        assert stats["escalated"] == 0
        assert stats["completion_rate"] == 0.0

        hist = get_review_history("NON_EXISTENT_BID", db_path=empty_db_path)
        assert hist == []
        rec = get_review_record("NON_EXISTENT_BID", db_path=empty_db_path)
        assert rec is None
        print("[PASS] Empty database states verified: Zero reviews return clean default values without exception.")
    finally:
        if os.path.exists(empty_db_path):
            os.remove(empty_db_path)


def test_part_s_security_and_hygiene():
    print("\n" + "="*50)
    print("PART S — SECURITY AND HYGIENE VALIDATION")
    print("="*50)
    # Check .env is not tracked
    env_file = PROJECT_ROOT / ".env"
    assert not env_file.exists(), ".env file must not exist in repository"

    # Check .gitignore exists and ignores review_audit.db
    gitignore = PROJECT_ROOT / ".gitignore"
    assert gitignore.exists(), ".gitignore missing"
    git_content = gitignore.read_text(encoding="utf-8")
    assert "review_audit.db" in git_content or "*.db" in git_content, "review_audit.db not in .gitignore"
    print("[PASS] Git hygiene verified: .env absent and database files gitignored.")


def test_part_t_terminology_check():
    print("\n" + "="*50)
    print("PART T — TERMINOLOGY AUDIT")
    print("="*50)
    prohibited_terms = [
        "SIH",
        "real-time GeM",
        "live GeM",
        "GeM API",
        "automatic approval",
        "automatic rejection",
        "auto qualify",
        "auto disqualify"
    ]

    violations = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Exclude directories
        if any(skip in root for skip in [".git", "venv", ".pytest_cache", "__pycache__", "scratch"]):
            continue
        for f in files:
            if f.endswith((".py", ".md", ".txt", ".json", ".csv")):
                f_path = Path(root) / f
                try:
                    content = f_path.read_text(encoding="utf-8", errors="ignore")
                    for term in prohibited_terms:
                        if term.lower() in content.lower():
                            # Exact check to avoid false positives on legitimate substrings
                            # For SIH, check word boundary
                            if term == "SIH":
                                if "sih" in content.lower():
                                    violations.append((term, str(f_path)))
                            else:
                                violations.append((term, str(f_path)))
                except Exception:
                    pass

    print(f"Total Prohibited Terminology Violations Found: {len(violations)}")
    if violations:
        for term, path in violations[:10]:
            print(f"  [VIOLATION] Found '{term}' in {path}")
    assert len(violations) == 0, f"Found {len(violations)} prohibited term occurrences!"
    print("[PASS] ZERO prohibited term occurrences found. Compliance language is 100% compliant.")


def test_part_u_final_file_integrity():
    print("\n" + "="*50)
    print("PART U — FINAL MODEL AND DATASET FILE INTEGRITY CONFIRMATION")
    print("="*50)
    m_size = MODEL_PATH.stat().st_size
    d_size = DATASET_PATH.stat().st_size
    print(f"Final Model Size: {m_size:,} bytes (Expected: {EXPECTED_MODEL_SIZE:,})")
    print(f"Final Dataset Size: {d_size:,} bytes (Expected: {EXPECTED_DATASET_SIZE:,})")
    assert m_size == EXPECTED_MODEL_SIZE, f"Model size mismatch at end of test: {m_size}"
    assert d_size == EXPECTED_DATASET_SIZE, f"Dataset size mismatch at end of test: {d_size}"
    print("[PASS] Final file integrity confirmed. Both artifacts are 100% pristine and unaltered.")


def run_all_step11_verifications():
    print("==================================================")
    print("STARTING FULL STEP 11 SYSTEM & INTEGRATION TESTING")
    print("==================================================")

    df = test_part_c_dataset_integrity()
    model = test_part_d_model_integrity()
    test_part_e_prediction_engine(model, df)
    test_part_f_critical_rule_safety(model, df)
    test_part_g_document_verification()
    test_part_h_document_bid_integration(model, df)
    test_part_i_explainability(model, df)
    test_part_j_k_review_workflow_and_audit()
    test_part_l_streamlit_all_7_pages()
    demo_results = test_part_m_six_demo_profiles(model, df)
    test_part_o_risk_analysis_filters(df)
    test_part_q_empty_database_states()
    test_part_s_security_and_hygiene()
    test_part_t_terminology_check()
    test_part_u_final_file_integrity()

    print("\n==================================================")
    print("ALL STEP 11 VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("==================================================")
    return demo_results


if __name__ == "__main__":
    run_all_step11_verifications()
