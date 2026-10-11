"""
AIML Hackathon Problem Statement ID: 26100
Organization: Chennai Petroleum Corporation Limited (CPCL)
Theme: Smart Automation

Test Suite: AI Explainability & Compliance Intelligence Module
File: tests/test_explainability.py

Verifies:
1. Explanation generated for compliant bid
2. Explanation generated for non-compliant bid
3. Positive factors generated
4. Risk factors generated
5. Rule-based findings integrated
6. Officer summary generated
7. Review priority generated
8. Missing/invalid input handled safely
"""

import pytest
import pandas as pd
from pathlib import Path

from src.explainability import (
    explain_prediction,
    get_top_positive_factors,
    get_top_risk_factors,
    get_review_priority,
    generate_officer_summary,
    generate_review_recommendations,
    load_feature_importance_table,
    EXPLAINABILITY_DISCLAIMER
)


@pytest.fixture(scope="module")
def dataset():
    """Loads the synthetic compliance dataset in read-only mode."""
    data_path = Path("data/bid_compliance_dataset.csv")
    assert data_path.exists(), "Dataset not found at data/bid_compliance_dataset.csv"
    return pd.read_csv(data_path)


@pytest.fixture(scope="module")
def compliant_bid(dataset):
    """Returns a known compliant bid sample."""
    sample = dataset[dataset["bid_id"] == "BID0001"]
    if not sample.empty:
        return sample.iloc[0]
    return dataset[dataset["actual_compliance"] == 1].iloc[0]


@pytest.fixture(scope="module")
def non_compliant_bid(dataset):
    """Returns a known non-compliant / high-risk bid sample."""
    sample = dataset[dataset["bid_id"] == "BID0073"]
    if not sample.empty:
        return sample.iloc[0]
    return dataset[dataset["actual_compliance"] == 0].iloc[0]


# -------------------------------------------------------------
# 1. COMPLIANT BID EXPLANATION TEST
# -------------------------------------------------------------
def test_explain_prediction_compliant(compliant_bid):
    """Validates explanation generation for a compliant bid."""
    res = explain_prediction(compliant_bid)
    assert res is not None
    assert res["prediction_outcome"] == 1
    assert res["prediction_label"] == "Compliant"
    assert res["risk_level"] == "Low"
    assert res["review_priority"] == "STANDARD REVIEW"
    assert len(res["positive_factors"]) > 0
    assert "contributed" in res["positive_factors"][0]["description"].lower()
    assert res["disclaimer"] == EXPLAINABILITY_DISCLAIMER


# -------------------------------------------------------------
# 2. NON-COMPLIANT BID EXPLANATION TEST
# -------------------------------------------------------------
def test_explain_prediction_non_compliant(non_compliant_bid):
    """Validates explanation generation for a non-compliant / excluded bid."""
    res = explain_prediction(non_compliant_bid)
    assert res is not None
    assert res["prediction_outcome"] == 0
    assert res["prediction_label"] == "Non-Compliant / Review"
    assert res["risk_level"] == "High"
    assert res["review_priority"] == "HIGH PRIORITY REVIEW"
    assert len(res["risk_factors"]) > 0
    assert any("blacklist" in r["factor"].lower() for r in res["risk_factors"])


# -------------------------------------------------------------
# 3. POSITIVE FACTORS GENERATION TEST
# -------------------------------------------------------------
def test_positive_factors_generation(compliant_bid):
    """Validates that only genuine positive factors are extracted and ranked."""
    fi_df = load_feature_importance_table()
    pos = get_top_positive_factors(compliant_bid, fi_df)
    assert len(pos) >= 5
    # Ensure sorted by importance descending
    importances = [p["importance"] for p in pos]
    assert importances == sorted(importances, reverse=True)

    factors_text = [p["factor"].lower() for p in pos]
    assert any("gst" in f for f in factors_text)
    assert any("pan" in f for f in factors_text)


# -------------------------------------------------------------
# 4. RISK FACTORS GENERATION TEST
# -------------------------------------------------------------
def test_risk_factors_generation(non_compliant_bid):
    """Validates that genuine risk indicators are identified and sorted."""
    fi_df = load_feature_importance_table()
    risks = get_top_risk_factors(non_compliant_bid, fi_df)
    assert len(risks) > 0
    importances = [r["importance"] for r in risks]
    assert importances == sorted(importances, reverse=True)

    risk_names = [r["factor"].lower() for r in risks]
    assert any("blacklist" in r for r in risk_names)


# -------------------------------------------------------------
# 5. RULE-BASED FINDINGS INTEGRATION TEST
# -------------------------------------------------------------
def test_rule_based_findings_integration(compliant_bid, non_compliant_bid):
    """Validates that rule-based checks are partitioned into passed, warnings, and failed."""
    res_comp = explain_prediction(compliant_bid)
    rules_comp = res_comp["rule_based_checks"]
    assert "passed" in rules_comp
    assert "warnings" in rules_comp
    assert "failed" in rules_comp
    assert len(rules_comp["passed"]) >= 15
    assert len(rules_comp["failed"]) == 0

    res_risk = explain_prediction(non_compliant_bid)
    rules_risk = res_risk["rule_based_checks"]
    assert len(rules_risk["failed"]) > 0
    assert any("blacklist" in f.lower() for f in rules_risk["failed"])


# -------------------------------------------------------------
# 6. OFFICER SUMMARY GENERATION TEST
# -------------------------------------------------------------
def test_officer_summary_generation(compliant_bid, non_compliant_bid):
    """Validates non-technical plain-language Procurement Officer summary."""
    res_comp = explain_prediction(compliant_bid)
    summary_comp = res_comp["officer_summary"]
    assert "AI assessment indicates" in summary_comp
    assert "lower-risk bid" in summary_comp
    assert "Standard officer verification is still required" in summary_comp
    # Must NOT claim definitive legal approval
    assert "automatically approved" not in summary_comp.lower()
    assert "officially compliant" not in summary_comp.lower()

    res_risk = explain_prediction(non_compliant_bid)
    summary_risk = res_risk["officer_summary"]
    assert "high-risk" in summary_risk.lower()
    assert "Procurement Officer" in summary_risk
    # Must NOT claim causal failure
    assert "caused the bid to fail" not in summary_risk.lower()


# -------------------------------------------------------------
# 7. REVIEW PRIORITY GENERATION TEST
# -------------------------------------------------------------
def test_review_priority_generation(compliant_bid, non_compliant_bid, dataset):
    """Validates review priority mapping across risk tiers."""
    # Low Risk -> STANDARD REVIEW
    p_comp = explain_prediction(compliant_bid)
    assert p_comp["review_priority"] == "STANDARD REVIEW"
    assert p_comp["urgency"] == "Routine"

    # High Risk with Blacklist -> HIGH PRIORITY REVIEW with exclusion banner
    p_risk = explain_prediction(non_compliant_bid)
    assert p_risk["review_priority"] == "HIGH PRIORITY REVIEW"
    assert "STATUTORY EXCLUSION" in p_risk["review_badge"]

    # Medium Risk sample
    med_sample = dataset[dataset["bid_id"] == "BID0008"]
    if not med_sample.empty:
        p_med = explain_prediction(med_sample.iloc[0])
        assert p_med["review_priority"] in ["PRIORITY REVIEW", "HIGH PRIORITY REVIEW"]


# -------------------------------------------------------------
# 8. MISSING / INVALID INPUT HANDLING TEST
# -------------------------------------------------------------
def test_missing_and_invalid_input_handling():
    """Validates that sparse or missing input values are handled safely without unhandled exceptions."""
    minimal_bid = {
        "project_type": "IT Services",
        "state": "Maharashtra",
        "district": "Mumbai City",
        "bid_value_lakhs": 25.0,
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
        "local_content_percentage": 50.0,
        "digilocker_verification": "Verified",
        "blacklist_status": "Clear",
        "debarment_status": "Not Debarred",
        "tender_eligibility": "Eligible",
        "required_documents": 5,
        "submitted_documents": 5,
        "document_completeness": 100.0,
        "documentation_consistency": 85.0,
        "previous_compliance_issues": 0,
        "portal_verification_issues": 0,
        "verification_delay_days": 2,
        "compliance_score": 80.0
    }
    res = explain_prediction(minimal_bid)
    assert res is not None
    assert "officer_summary" in res
    assert len(res["recommendations"]) > 0

    # With Document Verification Payload attached
    fake_doc_payload = {
        "required_documents": 5,
        "submitted_documents": 4,
        "document_completeness": 80.0,
        "documentation_consistency": 65.0,
        "consistency_status": "REVIEW REQUIRED"
    }
    res_doc = explain_prediction(minimal_bid, doc_payload=fake_doc_payload)
    assert res_doc["doc_verification_context"]["available"] is True
    assert "80.0%" in res_doc["doc_verification_context"]["explanation"]
    assert "80.0%" in res_doc["officer_summary"]
