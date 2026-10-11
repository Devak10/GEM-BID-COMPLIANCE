"""
AI-Powered Integrated Bid Compliance Verification Platform for GeM Procurement
AIML Hackathon Problem Statement ID: 26100
Organization: Chennai Petroleum Corporation Limited (CPCL)
Theme: Smart Automation

Unit and Integration Tests: Prediction Engine & Critical Statutory Rule Safety
File: tests/test_predict_compliance.py
"""

import os
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.predict_compliance import (
    load_model,
    predict_bid,
    predict_from_row,
    run_basic_compliance_checks,
    assess_bid
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = str(PROJECT_ROOT / "models" / "compliance_model.pkl")
DATASET_PATH = str(PROJECT_ROOT / "data" / "bid_compliance_dataset.csv")


@pytest.fixture(scope="module")
def loaded_model():
    """Provides the trained compliance model for test assertions."""
    assert os.path.exists(MODEL_PATH), f"Model not found at {MODEL_PATH}"
    return load_model(MODEL_PATH)


@pytest.fixture(scope="module")
def sample_dataset():
    """Provides the synthetic procurement dataset for test cases."""
    assert os.path.exists(DATASET_PATH), f"Dataset not found at {DATASET_PATH}"
    return pd.read_csv(DATASET_PATH)


def test_load_model_valid(loaded_model):
    """1. Test that the trained pipeline loads and contains scikit-learn Pipeline methods."""
    assert loaded_model is not None
    assert hasattr(loaded_model, "predict")
    assert hasattr(loaded_model, "predict_proba")


def test_load_model_missing_file():
    """2. Test that loading a non-existent model path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_model("non_existent_model_dir/fake_model.pkl")


def test_predict_bid_normal_compliant(loaded_model, sample_dataset):
    """3. Test prediction on known compliant bidder (BID0001)."""
    bid_row = sample_dataset[sample_dataset["bid_id"] == "BID0001"].iloc[0].to_dict()
    res = predict_bid(bid_row, model=loaded_model)

    assert "predicted_compliance" in res
    assert "compliance_probability" in res
    assert "non_compliance_probability" in res
    assert res["predicted_compliance"] == 1
    assert res["compliance_probability"] > 0.85
    assert np.isclose(res["compliance_probability"] + res["non_compliance_probability"], 1.0)
    assert res["risk_level"] == "Low"


def test_predict_bid_high_risk(loaded_model, sample_dataset):
    """4. Test prediction on known high-risk bidder (BID0007)."""
    bid_row = sample_dataset[sample_dataset["bid_id"] == "BID0007"].iloc[0].to_dict()
    res = predict_bid(bid_row, model=loaded_model)

    assert res["predicted_compliance"] == 0
    assert res["compliance_probability"] < 0.20
    assert res["risk_level"] == "High"
    assert len(res["risk_reasons"]) > 0


def test_predict_from_row_convenience(loaded_model, sample_dataset):
    """5. Test convenience wrapper predict_from_row accepting pandas Series."""
    series_row = sample_dataset.iloc[0]
    res = predict_from_row(series_row, model=loaded_model)
    assert isinstance(res, dict)
    assert "predicted_compliance" in res


def test_run_basic_compliance_checks_all_pass():
    """6. Test deterministic statutory rule checks with fully clean inputs."""
    clean_data = {
        "gst_registration": "Verified",
        "gst_return_filing": "Filed",
        "pan_verification": "Verified",
        "udyam_registration": "Verified",
        "income_tax_compliance": "Compliant",
        "startup_india_status": "Verified",
        "nsic_status": "Verified",
        "epfo_compliance": "Compliant",
        "esic_compliance": "Compliant",
        "oem_authorization": "Valid",
        "make_in_india_compliance": "Compliant",
        "local_content_percentage": 70.0,
        "digilocker_verification": "Verified",
        "blacklist_status": "Clear",
        "debarment_status": "Not Debarred",
        "tender_eligibility": "Eligible",
        "document_completeness": 100.0,
        "documentation_consistency": 90.0,
        "previous_compliance_issues": 0,
        "portal_verification_issues": 0,
        "verification_delay_days": 3,
        "compliance_score": 95.0
    }
    checks = run_basic_compliance_checks(clean_data)
    assert len(checks["failed_checks"]) == 0
    assert len(checks["passed_checks"]) >= 15

def test_run_basic_compliance_checks_with_failures():
    """7. Test deterministic rule checks detect defaults and flag warnings."""
    failing_data = {
        "gst_registration": "Not Verified",
        "gst_return_filing": "Not Filed",
        "pan_verification": "Not Verified",
        "epfo_compliance": "Non-Compliant",
        "esic_compliance": "Non-Compliant",
        "oem_authorization": "Invalid",
        "blacklist_status": "Blacklisted",
        "debarment_status": "Debarred",
        "tender_eligibility": "Not Eligible",
        "document_completeness": 40.0,
        "documentation_consistency": 30.0,
        "previous_compliance_issues": 3,
        "portal_verification_issues": 3,
        "verification_delay_days": 20
    }
    checks = run_basic_compliance_checks(failing_data)
    assert len(checks["failed_checks"]) >= 5
    assert any("Blacklist" in f or "BLACKLISTED" in f for f in checks["failed_checks"])
    assert any("DEBARRED" in f for f in checks["failed_checks"])
    assert any("NOT ELIGIBLE" in f for f in checks["failed_checks"])

def test_critical_statutory_exclusion_blacklisted(loaded_model, sample_dataset):
    """8. Test statutory safety: Blacklisted vendor must have exclusion flag and High Risk."""
    blacklisted_row = sample_dataset[sample_dataset["bid_id"] == "BID0073"].iloc[0].to_dict()
    assessment = assess_bid(blacklisted_row, model=loaded_model)

    assert assessment["procurement_review"]["statutory_exclusions_flagged"] is True
    assert any("Blacklisted" in flag for flag in assessment["procurement_review"]["exclusion_flags"])
    assert assessment["risk_assessment"]["risk_level"] == "High"
    assert assessment["procurement_review"]["required"] is True

def test_critical_statutory_exclusion_debarred(loaded_model, sample_dataset):
    """9. Test statutory safety: Debarred vendor (BID0033) must have exclusion flag and High Risk."""
    debarred_row = sample_dataset[sample_dataset["bid_id"] == "BID0033"].iloc[0].to_dict()
    assessment = assess_bid(debarred_row, model=loaded_model)

    assert assessment["procurement_review"]["statutory_exclusions_flagged"] is True
    assert any("Debarred" in flag for flag in assessment["procurement_review"]["exclusion_flags"])
    assert assessment["risk_assessment"]["risk_level"] == "High"
    assert assessment["procurement_review"]["required"] is True

def test_critical_statutory_exclusion_not_eligible(loaded_model, sample_dataset):
    """10. Test statutory safety: Ineligible bidder (BID0021) must have exclusion flag and High Risk."""
    ineligible_row = sample_dataset[sample_dataset["bid_id"] == "BID0021"].iloc[0].to_dict()
    assessment = assess_bid(ineligible_row, model=loaded_model)

    assert assessment["procurement_review"]["statutory_exclusions_flagged"] is True
    assert any("Not Eligible" in flag for flag in assessment["procurement_review"]["exclusion_flags"])
    assert assessment["risk_assessment"]["risk_level"] == "High"
    assert assessment["procurement_review"]["required"] is True

def test_high_ml_probability_with_statutory_exclusion_handling(loaded_model):
    """
    11. Critical Rule Safety Validation:
    Verify that an artificial bidder with high general compliance scores
    but with Debarred status is NOT marked Low Risk and has critical exclusion flagged.
    """
    synthetic_debarred = {
        "bid_id": "SYNTH_DEBARRED_01",
        "company_name": "High Prob Debarred Vendor Corp",
        "project_type": "IT Services",
        "state": "Maharashtra",
        "district": "Mumbai City",
        "bid_value_lakhs": 50.0,
        "gst_registration": "Verified",
        "gst_return_filing": "Filed",
        "pan_verification": "Verified",
        "udyam_registration": "Verified",
        "income_tax_compliance": "Compliant",
        "startup_india_status": "Not Applicable",
        "nsic_status": "Not Applicable",
        "epfo_compliance": "Compliant",
        "esic_compliance": "Compliant",
        "oem_authorization": "Valid",
        "make_in_india_compliance": "Compliant",
        "local_content_percentage": 80.0,
        "digilocker_verification": "Verified",
        "blacklist_status": "Clear",
        "debarment_status": "Debarred",  # Statutory exclusion
        "tender_eligibility": "Eligible",
        "required_documents": 10,
        "submitted_documents": 10,
        "document_completeness": 100.0,
        "documentation_consistency": 95.0,
        "previous_compliance_issues": 0,
        "portal_verification_issues": 0,
        "verification_delay_days": 2,
        "compliance_score": 90.0
    }
    assessment = assess_bid(synthetic_debarred, model=loaded_model)

    # 1. Rule layer MUST catch exclusion
    assert assessment["procurement_review"]["statutory_exclusions_flagged"] is True
    assert "Statutory Exclusion: Vendor is Debarred" in assessment["procurement_review"]["exclusion_flags"]
    # 2. Risk assessment MUST escalate to High Risk due to debarment
    assert assessment["risk_assessment"]["risk_level"] == "High"
    assert any("Debarment" in r for r in assessment["risk_assessment"]["reasons"])
    # 3. Mandatory human-in-the-loop review required
    assert assessment["procurement_review"]["required"] is True

def test_missing_and_invalid_values_resilience(loaded_model, sample_dataset):
    """12. Edge Case Handling: Missing schema columns raise ValueError cleanly."""
    empty_row = {
        "bid_id": "BID_EMPTY",
        "company_name": "Empty Test Co"
    }
    with pytest.raises(ValueError) as excinfo:
        predict_bid(empty_row, model=loaded_model)
    assert "missing required model features" in str(excinfo.value)

def test_unusual_categorical_values(loaded_model, sample_dataset):
    """13. Edge Case Handling: Unknown categorical values outside training vocabulary are ignored by OneHotEncoder."""
    base_row = sample_dataset.iloc[0].to_dict()
    unusual_row = dict(base_row)
    unusual_row.update({
        "project_type": "Aerospace Telemetry",
        "state": "Unknown State",
        "district": "Unknown District",
        "gst_registration": "Pending Verification 99",
        "blacklist_status": "Clear",
        "debarment_status": "Not Debarred"
    })
    assessment = assess_bid(unusual_row, model=loaded_model)
    assert assessment is not None
    assert assessment["prediction"]["predicted_compliance"] in [0, 1]
    assert 0.0 <= assessment["prediction"]["compliance_probability"] <= 1.0
