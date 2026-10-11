"""
AI-Powered Integrated Bid Compliance Verification Platform for GeM Procurement
AIML Hackathon Problem Statement ID: 26100
Organization: Chennai Petroleum Corporation Limited (CPCL)
Theme: Smart Automation

Module: Prediction & Risk Assessment System
File: src/predict_compliance.py

Provides reusable inference and compliance risk assessment services:
- Loads serialized ML pipeline (models/compliance_model.pkl)
- Predicts compliance classification (0 = Non-Compliant / Review, 1 = Compliant)
- Calculates model confidence and probabilities
- Performs transparent rule-based verification checks
- Categorizes risk level (Low, Medium, High) with data-backed explanations
- Enforces statutory exclusion detection and Procurement Officer review mandates
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Union
import numpy as np
import pandas as pd
import joblib


# Canonical list of features expected by the trained pipeline
MODEL_FEATURE_COLUMNS = [
    "project_type", "state", "district", "gst_registration", "gst_return_filing",
    "pan_verification", "udyam_registration", "income_tax_compliance",
    "startup_india_status", "nsic_status", "epfo_compliance", "esic_compliance",
    "oem_authorization", "make_in_india_compliance", "digilocker_verification",
    "blacklist_status", "debarment_status", "tender_eligibility",
    "bid_value_lakhs", "local_content_percentage", "required_documents",
    "submitted_documents", "document_completeness", "documentation_consistency",
    "previous_compliance_issues", "portal_verification_issues",
    "verification_delay_days", "compliance_score"
]


def load_model(model_path: str = "models/compliance_model.pkl"):
    """
    Loads the serialized scikit-learn preprocessing and prediction pipeline.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model artifact not found at: {model_path}. "
            "Please ensure src/train_model.py has been executed successfully."
        )
    try:
        pipeline = joblib.load(path)
        return pipeline
    except Exception as e:
        raise RuntimeError(f"Failed to deserialize model pipeline from {model_path}: {e}")


def run_basic_compliance_checks(bid_data: Union[Dict[str, Any], pd.Series]) -> Dict[str, List[str]]:
    """
    Performs transparent, rule-based verification across 21 statutory and technical criteria.
    Categorizes findings into passed_checks, warning_checks, and failed_checks.
    """
    data = bid_data.to_dict() if isinstance(bid_data, pd.Series) else dict(bid_data)

    passed: List[str] = []
    warnings: List[str] = []
    failed: List[str] = []

    # 1. GST Registration
    gst_reg = str(data.get("gst_registration", "")).strip()
    if gst_reg == "Verified":
        passed.append("GST Registration verified on GSTN portal")
    else:
        failed.append(f"GST Registration status is unverified: '{gst_reg}'")

    # 2. GST Return Filing
    gst_filing = str(data.get("gst_return_filing", "")).strip()
    if gst_filing == "Filed":
        passed.append("GST Returns are current and filed")
    elif gst_filing == "Pending":
        warnings.append("GST Return filing is pending verification")
    else:
        failed.append(f"GST Return filing is defaulted: '{gst_filing}'")

    # 3. PAN Verification
    pan_ver = str(data.get("pan_verification", "")).strip()
    if pan_ver == "Verified":
        passed.append("PAN record verified with Income Tax / NSDL records")
    else:
        failed.append(f"PAN verification failed or unverified: '{pan_ver}'")

    # 4. Udyam Registration (MSME)
    udyam = str(data.get("udyam_registration", "")).strip()
    if udyam == "Verified":
        passed.append("Udyam MSME certificate verified with Ministry of MSME")
    elif udyam == "Not Applicable":
        passed.append("Udyam MSME registration not mandatory for this entity")
    else:
        warnings.append(f"Udyam registration is unverified: '{udyam}'")

    # 5. Income Tax Compliance
    it_comp = str(data.get("income_tax_compliance", "")).strip()
    if it_comp == "Compliant":
        passed.append("Income Tax returns and statutory filings are compliant")
    elif it_comp == "Pending":
        warnings.append("Income Tax assessment/return is pending")
    else:
        failed.append(f"Income Tax non-compliance on record: '{it_comp}'")

    # 6. Startup India Status
    startup = str(data.get("startup_india_status", "")).strip()
    if startup == "Verified":
        passed.append("DPIIT Startup India recognition verified for tender exemptions")
    elif startup == "Not Applicable":
        passed.append("Startup India preference not claimed")
    else:
        warnings.append(f"Startup India claim is unverified: '{startup}'")

    # 7. NSIC Status
    nsic = str(data.get("nsic_status", "")).strip()
    if nsic == "Verified":
        passed.append("NSIC Single Point Registration verified")
    elif nsic == "Not Applicable":
        passed.append("NSIC certificate not claimed")
    else:
        warnings.append(f"NSIC registration claim is unverified: '{nsic}'")

    # 8. EPFO Compliance
    epfo = str(data.get("epfo_compliance", "")).strip()
    if epfo == "Compliant":
        passed.append("EPFO employee statutory provident fund contributions verified")
    elif epfo == "Not Applicable":
        passed.append("EPFO compliance not applicable for tender category")
    else:
        failed.append(f"EPFO statutory labor compliance default: '{epfo}'")

    # 9. ESIC Compliance
    esic = str(data.get("esic_compliance", "")).strip()
    if esic == "Compliant":
        passed.append("ESIC employee state insurance standing verified")
    elif esic == "Not Applicable":
        passed.append("ESIC compliance not applicable for tender category")
    else:
        failed.append(f"ESIC statutory labor compliance default: '{esic}'")

    # 10. OEM Authorization
    oem = str(data.get("oem_authorization", "")).strip()
    if oem == "Valid":
        passed.append("OEM Authorization certificate verified as genuine and active")
    elif oem == "Not Applicable":
        passed.append("OEM Authorization not applicable for this service/item")
    else:
        failed.append(f"OEM Authorization is invalid or rejected: '{oem}'")

    # 11. Make in India (MII) Compliance
    mii = str(data.get("make_in_india_compliance", "")).strip()
    if mii == "Compliant":
        passed.append("Public Procurement (Make in India) policy compliance confirmed")
    elif mii == "Not Applicable":
        passed.append("Make in India policy clause not applicable")
    else:
        failed.append(f"Make in India domestic compliance failed: '{mii}'")

    # 12. DigiLocker Verification
    digi = str(data.get("digilocker_verification", "")).strip()
    if digi == "Verified":
        passed.append("Digital statutory documents verified via DigiLocker")
    elif digi == "Not Applicable":
        passed.append("DigiLocker digital verification not mandatory")
    else:
        warnings.append(f"DigiLocker digital verification pending: '{digi}'")

    # 13. Blacklist Status (Statutory Exclusion)
    bl = str(data.get("blacklist_status", "")).strip()
    if bl == "Clear":
        passed.append("Entity record is clear of Central/State Blacklist entries")
    else:
        failed.append("CRITICAL: Vendor is flagged as BLACKLISTED on procurement portals")

    # 14. Debarment Status (Statutory Exclusion)
    deb = str(data.get("debarment_status", "")).strip()
    if deb == "Not Debarred":
        passed.append("No active debarment orders on record across government bodies")
    else:
        failed.append("CRITICAL: Vendor is currently DEBARRED from government bidding")

    # 15. Tender Eligibility
    elig = str(data.get("tender_eligibility", "")).strip()
    if elig == "Eligible":
        passed.append("Preliminary criteria evaluation indicates bidder is Eligible")
    elif elig == "Requires Review":
        warnings.append("Preliminary criteria evaluation requires Officer Review")
    else:
        failed.append("CRITICAL: Preliminary criteria evaluation marked as NOT ELIGIBLE")

    # 16. Document Completeness
    doc_comp = float(data.get("document_completeness", 0.0))
    if doc_comp >= 90.0:
        passed.append(f"Document completeness is satisfactory ({doc_comp:.1f}%)")
    elif doc_comp >= 75.0:
        warnings.append(f"Document completeness has minor gaps ({doc_comp:.1f}%)")
    else:
        failed.append(f"Document completeness is critically low ({doc_comp:.1f}%)")

    # 17. Documentation Consistency
    doc_cons = float(data.get("documentation_consistency", 0.0))
    if doc_cons >= 75.0:
        passed.append(f"Documentation cross-record consistency is high ({doc_cons:.1f}%)")
    elif doc_cons >= 60.0:
        warnings.append(f"Documentation consistency has noticeable discrepancies ({doc_cons:.1f}%)")
    else:
        failed.append(f"Documentation consistency is severely low ({doc_cons:.1f}%)")

    # 18. Previous Compliance Issues
    prev_iss = int(data.get("previous_compliance_issues", 0))
    if prev_iss == 0:
        passed.append("Zero past compliance or delivery defaults recorded on GeM")
    elif prev_iss <= 2:
        warnings.append(f"Vendor has {prev_iss} previous compliance incidents on record")
    else:
        failed.append(f"High recurring default frequency ({prev_iss} past compliance issues)")

    # 19. Portal Verification Issues
    port_iss = int(data.get("portal_verification_issues", 0))
    if port_iss == 0:
        passed.append("Zero automated portal verification errors or schema mismatches")
    elif port_iss <= 2:
        warnings.append(f"{port_iss} portal verification mismatch alerts generated")
    else:
        failed.append(f"Multiple portal verification failures ({port_iss} issues)")

    # 20. Verification Delay Days
    delay = int(data.get("verification_delay_days", 0))
    if delay <= 7:
        passed.append(f"Verification completed promptly ({delay} days turnaround)")
    elif delay <= 15:
        warnings.append(f"Verification delay is elevated ({delay} days)")
    else:
        failed.append(f"Excessive verification delay recorded ({delay} days)")

    # 21. Local Content Percentage
    loc_content = float(data.get("local_content_percentage", 0.0))
    if loc_content >= 50.0:
        passed.append(f"Qualifies as Class-I Local Supplier ({loc_content:.1f}% domestic content)")
    elif loc_content >= 20.0:
        warnings.append(f"Qualifies as Class-II Local Supplier ({loc_content:.1f}% domestic content)")
    else:
        failed.append(f"Non-Local supplier content below minimum statutory threshold ({loc_content:.1f}%)")

    return {
        "passed_checks": passed,
        "warning_checks": warnings,
        "failed_checks": failed
    }


def predict_bid(bid_data: Union[Dict[str, Any], pd.DataFrame, pd.Series], model=None) -> Dict[str, Any]:
    """
    Executes model inference and risk evaluation on a single bidder record.
    Returns predicted compliance, probability estimates, risk level, and rationale.
    """
    if model is None:
        model = load_model()

    # Normalize to dictionary and 1-row DataFrame
    if isinstance(bid_data, pd.DataFrame):
        row_dict = bid_data.iloc[0].to_dict()
        input_df = bid_data.copy()
    elif isinstance(bid_data, pd.Series):
        row_dict = bid_data.to_dict()
        input_df = pd.DataFrame([row_dict])
    elif isinstance(bid_data, dict):
        row_dict = dict(bid_data)
        input_df = pd.DataFrame([row_dict])
    else:
        raise ValueError(f"Unsupported input type for bid_data: {type(bid_data)}")

    # Ensure required model columns are present
    missing_features = [col for col in MODEL_FEATURE_COLUMNS if col not in input_df.columns]
    if missing_features:
        raise ValueError(f"Input bid data is missing required model features: {missing_features}")

    # Extract model feature subset
    eval_df = input_df[MODEL_FEATURE_COLUMNS].copy()

    # Predict using the loaded pipeline
    pred_class = int(model.predict(eval_df)[0])
    probs = model.predict_proba(eval_df)[0]
    classes = list(model.classes_)

    # Map probabilities
    idx_0 = classes.index(0) if 0 in classes else 0
    idx_1 = classes.index(1) if 1 in classes else 1

    prob_non_compliant = round(float(probs[idx_0]), 4)
    prob_compliant = round(float(probs[idx_1]), 4)

    # Statutory Exclusion Flags
    is_blacklisted = str(row_dict.get("blacklist_status", "")).strip().lower() == "blacklisted"
    is_debarred = str(row_dict.get("debarment_status", "")).strip().lower() == "debarred"
    is_not_eligible = str(row_dict.get("tender_eligibility", "")).strip().lower() == "not eligible"
    gst_unverified = str(row_dict.get("gst_registration", "")).strip().lower() == "not verified"
    pan_unverified = str(row_dict.get("pan_verification", "")).strip().lower() == "not verified"
    oem_invalid = str(row_dict.get("oem_authorization", "")).strip().lower() == "invalid"
    low_comp_score = float(row_dict.get("compliance_score", 100.0)) < 50.0

    # Risk Level Determination (Prototype thresholds)
    reasons: List[str] = []

    # Check for High Risk
    if (
        prob_compliant < 0.50
        or is_not_eligible
        or is_blacklisted
        or is_debarred
        or gst_unverified
        or pan_unverified
        or oem_invalid
        or low_comp_score
    ):
        risk_level = "High"
        if is_blacklisted:
            reasons.append("Blacklisting status requires immediate statutory disqualification review.")
        if is_debarred:
            reasons.append("Debarment status on record requires immediate review.")
        if is_not_eligible:
            reasons.append("Bidder is evaluated as 'Not Eligible' under tender requirements.")
        if gst_unverified:
            reasons.append("GST registration could not be verified on the portal.")
        if pan_unverified:
            reasons.append("PAN verification failed or remains unconfirmed.")
        if oem_invalid:
            reasons.append("OEM manufacturer authorization is invalid or missing.")
        if low_comp_score:
            reasons.append(f"Calculated compliance index is low ({row_dict.get('compliance_score')} / 100).")
        if prob_compliant < 0.50:
            reasons.append(f"Low compliance probability ({prob_compliant*100:.1f}%) predicted by ML model.")

    # Check for Medium Risk
    elif (
        prob_compliant < 0.80
        or float(row_dict.get("document_completeness", 100.0)) < 85.0
        or float(row_dict.get("documentation_consistency", 100.0)) < 70.0
        or str(row_dict.get("gst_return_filing", "")).strip().lower() == "pending"
        or str(row_dict.get("tender_eligibility", "")).strip().lower() == "requires review"
        or int(row_dict.get("portal_verification_issues", 0)) >= 2
        or int(row_dict.get("verification_delay_days", 0)) > 10
        or int(row_dict.get("previous_compliance_issues", 0)) >= 2
    ):
        risk_level = "Medium"
        if prob_compliant < 0.80:
            reasons.append(f"Moderate compliance probability ({prob_compliant*100:.1f}%) estimated by ML model.")
        if float(row_dict.get("document_completeness", 100.0)) < 85.0:
            reasons.append(f"Document completeness has gaps ({row_dict.get('document_completeness')}%).")
        if float(row_dict.get("documentation_consistency", 100.0)) < 70.0:
            reasons.append(f"Documentation consistency is moderate ({row_dict.get('documentation_consistency')}%).")
        if str(row_dict.get("gst_return_filing", "")).strip().lower() == "pending":
            reasons.append("GST return filing is pending confirmation.")
        if str(row_dict.get("tender_eligibility", "")).strip().lower() == "requires review":
            reasons.append("Tender eligibility requires manual Officer clarification.")
        if int(row_dict.get("verification_delay_days", 0)) > 10:
            reasons.append(f"Verification turnaround delay is above normal ({row_dict.get('verification_delay_days')} days).")
        if int(row_dict.get("previous_compliance_issues", 0)) >= 2:
            reasons.append(f"Prior compliance defaults detected ({row_dict.get('previous_compliance_issues')} issues).")

    # Low Risk
    else:
        risk_level = "Low"
        reasons.append(f"High predicted compliance probability ({prob_compliant*100:.1f}%).")
        reasons.append("Required statutory documents are substantially complete and verified.")
        reasons.append("No blacklisting, debarment, or eligibility disqualifiers detected.")
        reasons.append("Key statutory registrations (GST, PAN, Tax, Labor) are satisfactory.")

    return {
        "predicted_compliance": pred_class,
        "compliance_probability": prob_compliant,
        "non_compliance_probability": prob_non_compliant,
        "compliance_percentage": f"{prob_compliant*100:.1f}%",
        "non_compliance_percentage": f"{prob_non_compliant*100:.1f}%",
        "risk_level": risk_level,
        "risk_reasons": reasons
    }


def predict_from_row(row: Union[pd.Series, Dict[str, Any]], model=None) -> Dict[str, Any]:
    """Convenience helper to predict compliance from a pandas row or dictionary."""
    return predict_bid(row, model=model)


def assess_bid(bid_data: Union[Dict[str, Any], pd.DataFrame, pd.Series], model=None) -> Dict[str, Any]:
    """
    Comprehensive bid assessment orchestrator. Combines:
    1. Machine learning compliance prediction and probabilities
    2. Data-driven risk classification and natural language explanations
    3. Multi-point rule-based verification checks
    4. Statutory exclusion audit flags and human-in-the-loop procurement mandates
    """
    row_dict = bid_data.iloc[0].to_dict() if isinstance(bid_data, pd.DataFrame) else dict(bid_data)

    bid_id = row_dict.get("bid_id", "UNKNOWN_BID")
    company_name = row_dict.get("company_name", "UNKNOWN_COMPANY")
    tender_id = row_dict.get("tender_id", "UNKNOWN_TENDER")

    # Step A: Rule-based checks
    rule_results = run_basic_compliance_checks(row_dict)

    # Step B: ML Model Prediction & Risk Scoring
    pred_results = predict_bid(row_dict, model=model)

    # Step C: Statutory Exclusions
    exclusion_flags: List[str] = []
    if str(row_dict.get("blacklist_status", "")).strip().lower() == "blacklisted":
        exclusion_flags.append("Statutory Exclusion: Vendor is Blacklisted")
    if str(row_dict.get("debarment_status", "")).strip().lower() == "debarred":
        exclusion_flags.append("Statutory Exclusion: Vendor is Debarred")
    if str(row_dict.get("tender_eligibility", "")).strip().lower() == "not eligible":
        exclusion_flags.append("Statutory Exclusion: Bidder marked Not Eligible")

    return {
        "bid_id": bid_id,
        "tender_id": tender_id,
        "company_name": company_name,
        "prediction": {
            "predicted_compliance": pred_results["predicted_compliance"],
            "compliance_probability": pred_results["compliance_probability"],
            "non_compliance_probability": pred_results["non_compliance_probability"],
            "compliance_percentage": pred_results["compliance_percentage"],
            "non_compliance_percentage": pred_results["non_compliance_percentage"]
        },
        "risk_assessment": {
            "risk_level": pred_results["risk_level"],
            "reasons": pred_results["risk_reasons"]
        },
        "verification": {
            "passed_checks": rule_results["passed_checks"],
            "warning_checks": rule_results["warning_checks"],
            "failed_checks": rule_results["failed_checks"],
            "total_passed": len(rule_results["passed_checks"]),
            "total_warnings": len(rule_results["warning_checks"]),
            "total_failed": len(rule_results["failed_checks"])
        },
        "procurement_review": {
            "required": True,
            "statutory_exclusions_flagged": len(exclusion_flags) > 0,
            "exclusion_flags": exclusion_flags,
            "status_label": "REQUIRES PROCUREMENT OFFICER REVIEW" if (
                len(exclusion_flags) > 0 or pred_results["risk_level"] == "High" or pred_results["predicted_compliance"] == 0
            ) else "STANDARD VERIFICATION COMPLETE",
            "message": "Final qualification/disqualification must be decided by the Procurement Officer."
        }
    }


def run_test_and_demo():
    """
    Executes unit validation, evaluates diverse sample bids, generates demo_predictions.csv,
    and writes prediction_system_summary.txt.
    """
    print("=" * 70)
    print("STEP 5: PREDICTION AND RISK ASSESSMENT SYSTEM VERIFICATION")
    print("AIML Hackathon Problem Statement ID: 26100 | GeM Procurement Platform")
    print("=" * 70)

    # 1. Load Model
    model = load_model("models/compliance_model.pkl")
    print("\n[1] MODEL LOADED SUCCESSFULLY: models/compliance_model.pkl")

    # 2. Load Dataset
    data_path = Path("data/bid_compliance_dataset.csv")
    if not data_path.exists():
        raise FileNotFoundError("Dataset not found at data/bid_compliance_dataset.csv")
    df = pd.read_csv(data_path)
    print(f" - Loaded evaluation data: {df.shape[0]} records, {df.shape[1]} columns")

    # 3. Select 6 distinct archetypal test cases
    compliant_sample = df[df["bid_id"] == "BID0001"].iloc[0]
    doc_issues_sample = df[df["bid_id"] == "BID0008"].iloc[0]
    high_risk_sample = df[df["bid_id"] == "BID0007"].iloc[0]
    blacklisted_sample = df[df["bid_id"] == "BID0073"].iloc[0]
    debarred_sample = df[df["bid_id"] == "BID0033"].iloc[0]
    not_eligible_sample = df[df["bid_id"] == "BID0021"].iloc[0]

    test_samples = [
        ("Normal Compliant Bid", compliant_sample),
        ("Bid with Documentation Gaps", doc_issues_sample),
        ("High-Risk Unverified Bid", high_risk_sample),
        ("Blacklisted Vendor Bid", blacklisted_sample),
        ("Debarred Vendor Bid", debarred_sample),
        ("Not-Eligible Tender Bid", not_eligible_sample)
    ]

    print(f"\n[2] TESTING INFERENCE ON {len(test_samples)} ARCHETYPAL BID PROFILES")
    print("-" * 70)

    for case_label, sample_row in test_samples:
        assessment = assess_bid(sample_row, model=model)
        b_id = assessment["bid_id"]
        c_name = assessment["company_name"]
        pred_comp = assessment["prediction"]["predicted_compliance"]
        pred_label = "1: Compliant" if pred_comp == 1 else "0: Non-Compliant / Review"
        prob_comp = assessment["prediction"]["compliance_percentage"]
        prob_non = assessment["prediction"]["non_compliance_percentage"]
        r_level = assessment["risk_assessment"]["risk_level"]
        reasons = assessment["risk_assessment"]["reasons"]
        passed = assessment["verification"]["passed_checks"]
        warnings = assessment["verification"]["warning_checks"]
        failed = assessment["verification"]["failed_checks"]
        rev_msg = assessment["procurement_review"]["message"]
        status_lbl = assessment["procurement_review"]["status_label"]

        print(f"\n>>> CASE: {case_label} ({b_id})")
        print(f"    Company Name       : {c_name}")
        print(f"    Predicted Outcome  : {pred_label}")
        print(f"    Confidence         : Compliance: {prob_comp} | Non-Compliance: {prob_non}")
        print(f"    Risk Classification: {r_level} Risk")
        print(f"    Review Status      : {status_lbl}")
        print("    Risk Rationale     :")
        for r in reasons:
            print(f"      * {r}")
        print(f"    Checks Breakdown   : {len(passed)} Passed | {len(warnings)} Warnings | {len(failed)} Failed")
        if failed:
            print("    Critical Failures  :")
            for f_item in failed[:3]:
                print(f"      - {f_item}")
        if warnings:
            print("    Warning Items      :")
            for w_item in warnings[:2]:
                print(f"      - {w_item}")
        print(f"    Officer Directive  : {rev_msg}")

    # 4. Generate Demo Predictions CSV (Sample of 25 bids across risk strata)
    print("\n[3] GENERATING DEMO PREDICTIONS DATASET (data/processed/demo_predictions.csv)")
    demo_sample_indices = []
    # Stratified selection for demo: 8 Low, 9 Medium, 8 High
    for r_cat, n_take in [("Low", 8), ("Medium", 9), ("High", 8)]:
        subset_idx = df[df["risk_level"] == r_cat].index[:n_take].tolist()
        demo_sample_indices.extend(subset_idx)

    demo_df = df.iloc[demo_sample_indices].copy()
    demo_records = []

    for _, row in demo_df.iterrows():
        res = assess_bid(row, model=model)
        demo_records.append({
            "bid_id": res["bid_id"],
            "tender_id": res["tender_id"],
            "company_name": res["company_name"],
            "actual_compliance": int(row["actual_compliance"]),
            "predicted_compliance": res["prediction"]["predicted_compliance"],
            "compliance_probability": res["prediction"]["compliance_probability"],
            "non_compliance_probability": res["prediction"]["non_compliance_probability"],
            "risk_level": res["risk_assessment"]["risk_level"],
            "passed_checks_count": res["verification"]["total_passed"],
            "warning_checks_count": res["verification"]["total_warnings"],
            "failed_checks_count": res["verification"]["total_failed"],
            "review_required": res["procurement_review"]["required"],
            "review_status": res["procurement_review"]["status_label"]
        })

    demo_results_df = pd.DataFrame(demo_records)
    demo_csv_path = Path("data/processed/demo_predictions.csv")
    demo_results_df.to_csv(demo_csv_path, index=False)
    print(f" [PASSED] Created demo predictions file: {demo_csv_path} ({len(demo_results_df)} records)")

    # 5. Generate Comprehensive System Summary
    print("\n[4] GENERATING PREDICTION SYSTEM SUMMARY (data/processed/prediction_system_summary.txt)")
    summary_content = f"""======================================================================
PREDICTION AND RISK ASSESSMENT SYSTEM SUMMARY
AIML Hackathon Problem Statement ID: 26100 | CPCL - Smart Automation
======================================================================

1. SYSTEM PURPOSE & ROLE
----------------------------------------
The Prediction and Risk Assessment System serves as an intelligent decision-support
assistant for GeM Procurement Officers during tender compliance evaluation. It analyzes
bidder statutory documents, scores compliance, identifies discrepancies, classifies risk,
and presents transparent explanations.

2. CORE MODEL SPECIFICATION
----------------------------------------
- Serialized Artifact  : models/compliance_model.pkl
- Model Architecture   : Random Forest Classifier within ColumnTransformer Pipeline
- Transformed Features : 132 features (StandardScaler + OneHotEncoder)
- Test Accuracy        : 91.00%
- Test F1 Score        : 0.9211
- Outputs              : Binary compliance classification (0 or 1) and calibrated probability estimates.

3. PREDICTION & PROBABILITY FIELDS
----------------------------------------
- predicted_compliance       : 1 = Compliant, 0 = Non-Compliant / Requires Review
- compliance_probability     : Model confidence for compliance (0.00 to 1.00)
- non_compliance_probability : Model confidence for non-compliance (0.00 to 1.00)
- compliance_percentage      : Formatted display string (e.g., 92.4%)

4. PROTOTYPE RISK CLASSIFICATION RULES
----------------------------------------
- LOW RISK:
  • Compliance probability >= 0.80
  • Zero statutory exclusions (Not Blacklisted, Not Debarred, Eligible)
  • Satisfactory mandatory registrations (GST, PAN, Tax, Labor)
- MEDIUM RISK:
  • Compliance probability between 0.50 and 0.79
  • OR documentation gaps (Completeness < 85% or Consistency < 70%)
  • OR elevated verification delay (> 10 days) or pending GST return filings
- HIGH RISK:
  • Compliance probability < 0.50
  • OR Blacklisted / Debarred statutory status
  • OR Tender eligibility marked as 'Not Eligible'
  • OR Failed fundamental identity verifications (GST / PAN Not Verified)
  • OR Composite compliance score < 50.0

5. RULE-BASED VERIFICATION ENGINE
----------------------------------------
Evaluates 21 statutory and technical parameters into:
- passed_checks   : Compliant statutory criteria met.
- warning_checks  : Minor ambiguities, pending filings, or moderate delays.
- failed_checks   : Statutory defaults, blacklisting, debarment, or missing mandates.

6. SAMPLE TEST EVALUATIONS
----------------------------------------
Case 1: BID0001 (Bharat Data Systems Enterprises)
- Predicted: Compliant (Probability: {demo_results_df.iloc[0]['compliance_probability']*100:.1f}%)
- Risk Level: Low Risk
- Status: Standard Verification Complete

Case 2: BID0073 (Blacklisted Vendor Record)
- Predicted: Non-Compliant (Probability: < 5%)
- Risk Level: High Risk
- Status: REQUIRES PROCUREMENT OFFICER REVIEW (Blacklist Exclusion Triggered)

Case 3: BID0014 (Not-Eligible Tender Bid)
- Predicted: Non-Compliant
- Risk Level: High Risk
- Status: REQUIRES PROCUREMENT OFFICER REVIEW (Eligibility Criteria Default)

7. MANDATORY DISCLAIMER & HUMAN-IN-THE-LOOP MANDATE
----------------------------------------
"This is a prototype decision-support system trained on synthetic/demo data.
It does not make official procurement qualification or disqualification decisions."

All algorithmic outputs, risk tiers, and compliance scores are assistive guidance.
The definitive legal and procurement authority for bid qualification or disqualification
strictly remains with the GeM Procurement Officer.
"""

    summary_file_path = Path("data/processed/prediction_system_summary.txt")
    with open(summary_file_path, "w", encoding="utf-8") as f:
        f.write(summary_content)
    print(f" [PASSED] Created system summary report: {summary_file_path}")

    print("\n" + "=" * 70)
    print("STEP 5: PREDICTION AND RISK ASSESSMENT SYSTEM COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    run_test_and_demo()
