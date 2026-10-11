"""
AI-Powered Integrated Bid Compliance Verification Platform for GeM Procurement
AIML Hackathon Problem Statement ID: 26100
Organization: Chennai Petroleum Corporation Limited (CPCL)
Theme: Smart Automation

Module: AI-Assisted Compliance Intelligence & Explainability Layer
File: src/explainability.py

Provides transparent decision-support explainability for procurement officers:
- Bid-specific explanation of model compliance prediction
- Decomposition into positive compliance indicators vs. risk indicators
- Integration of rule-based verification findings
- Transparent review priority assignment (Standard, Priority, High Priority)
- Plain-language Procurement Officer summary and actionable review directives
- Grounding in model feature importance without claiming causal relationships
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd

from src.predict_compliance import (
    load_model,
    assess_bid,
    run_basic_compliance_checks,
    MODEL_FEATURE_COLUMNS
)

# Canonical fallback feature importances based on trained Random Forest artifact
DEFAULT_FEATURE_IMPORTANCES: Dict[str, float] = {
    "compliance_score": 0.1785,
    "verification_delay_days": 0.1017,
    "tender_eligibility": 0.0880,
    "documentation_consistency": 0.0727,
    "document_completeness": 0.0365,
    "portal_verification_issues": 0.0330,
    "gst_registration": 0.0321,
    "local_content_percentage": 0.0259,
    "epfo_compliance": 0.0186,
    "esic_compliance": 0.0184,
    "previous_compliance_issues": 0.0183,
    "oem_authorization": 0.0159,
    "bid_value_lakhs": 0.0155,
    "pan_verification": 0.0143,
    "udyam_registration": 0.0137,
    "gst_return_filing": 0.0109,
    "make_in_india_compliance": 0.0095,
    "income_tax_compliance": 0.0092,
    "digilocker_verification": 0.0088,
    "blacklist_status": 0.0075,
    "debarment_status": 0.0068,
    "startup_india_status": 0.0055,
    "nsic_status": 0.0048
}

EXPLAINABILITY_DISCLAIMER = (
    "Explainability shows model signals and rule-based findings for decision support. "
    "Feature importance does not imply causation. Prototype rules and predictions require human verification."
)


def load_feature_importance_table(filepath: str = "data/processed/feature_importance.csv") -> pd.DataFrame:
    """
    Loads precomputed feature importance rankings from data/processed/feature_importance.csv.
    Falls back to a constructed DataFrame if the file is unavailable.
    """
    path = Path(filepath)
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass

    rows = [{"Feature": f"num__{k}", "Importance": v} for k, v in DEFAULT_FEATURE_IMPORTANCES.items()]
    return pd.DataFrame(rows)


def get_feature_weight(feature_name: str, fi_df: Optional[pd.DataFrame] = None) -> float:
    """
    Retrieves the importance weight for a base feature from the importance DataFrame or defaults.
    """
    if fi_df is not None and not fi_df.empty:
        # Search for any feature starting with num__<feature> or cat__<feature>
        matches = fi_df[fi_df["Feature"].str.contains(feature_name, case=False, regex=False)]
        if not matches.empty:
            return float(matches["Importance"].max())

    return DEFAULT_FEATURE_IMPORTANCES.get(feature_name, 0.01)


def get_top_positive_factors(
    bid_data: Union[Dict[str, Any], pd.Series],
    fi_df: Optional[pd.DataFrame] = None
) -> List[Dict[str, Any]]:
    """
    Identifies positive compliance indicators genuinely present in the submitted bid.
    Returns factors sorted by model importance weight.
    Only shows factors that are actually present in the submitted bid.
    """
    row = bid_data.to_dict() if isinstance(bid_data, pd.Series) else dict(bid_data)
    positive: List[Dict[str, Any]] = []

    # 1. Statutory Identity & Tax
    if str(row.get("gst_registration", "")).strip().lower() == "verified":
        w = get_feature_weight("gst_registration", fi_df)
        positive.append({
            "factor": "GST registration verified",
            "feature": "gst_registration",
            "value": "Verified",
            "importance": w,
            "description": "Active GSTIN registration verified on portal contributed positively to compliance confidence."
        })

    if str(row.get("gst_return_filing", "")).strip().lower() == "filed":
        w = get_feature_weight("gst_return_filing", fi_df)
        positive.append({
            "factor": "GST return filings current",
            "feature": "gst_return_filing",
            "value": "Filed",
            "importance": w,
            "description": "Timely GST filing history indicates regular statutory tax discipline."
        })

    if str(row.get("pan_verification", "")).strip().lower() == "verified":
        w = get_feature_weight("pan_verification", fi_df)
        positive.append({
            "factor": "PAN record verified",
            "feature": "pan_verification",
            "value": "Verified",
            "importance": w,
            "description": "Permanent Account Number confirmed against Income Tax / NSDL records."
        })

    if str(row.get("income_tax_compliance", "")).strip().lower() == "compliant":
        w = get_feature_weight("income_tax_compliance", fi_df)
        positive.append({
            "factor": "Income Tax filings compliant",
            "feature": "income_tax_compliance",
            "value": "Compliant",
            "importance": w,
            "description": "Statutory direct tax returns and assessments are compliant on record."
        })

    # 2. MSME & Preferences
    if str(row.get("udyam_registration", "")).strip().lower() == "verified":
        w = get_feature_weight("udyam_registration", fi_df)
        positive.append({
            "factor": "Udyam MSME registration verified",
            "feature": "udyam_registration",
            "value": "Verified",
            "importance": w,
            "description": "Official Ministry of MSME registration confirmed on portal."
        })

    # 3. Labor & Technical Mandates
    if str(row.get("epfo_compliance", "")).strip().lower() == "compliant":
        w = get_feature_weight("epfo_compliance", fi_df)
        positive.append({
            "factor": "EPFO statutory labor compliance satisfied",
            "feature": "epfo_compliance",
            "value": "Compliant",
            "importance": w,
            "description": "Provident fund contributions for workforce verified without statutory default."
        })

    if str(row.get("esic_compliance", "")).strip().lower() == "compliant":
        w = get_feature_weight("esic_compliance", fi_df)
        positive.append({
            "factor": "ESIC employee insurance standing compliant",
            "feature": "esic_compliance",
            "value": "Compliant",
            "importance": w,
            "description": "State insurance contributions verified without statutory default."
        })

    if str(row.get("oem_authorization", "")).strip().lower() == "valid":
        w = get_feature_weight("oem_authorization", fi_df)
        positive.append({
            "factor": "Valid OEM manufacturer authorization",
            "feature": "oem_authorization",
            "value": "Valid",
            "importance": w,
            "description": "Direct manufacturer authorization verified as active and genuine."
        })

    if str(row.get("make_in_india_compliance", "")).strip().lower() == "compliant":
        w = get_feature_weight("make_in_india_compliance", fi_df)
        positive.append({
            "factor": "Make in India policy compliance confirmed",
            "feature": "make_in_india_compliance",
            "value": "Compliant",
            "importance": w,
            "description": "Supplies meet domestic procurement requirements under Public Procurement policy."
        })

    # 4. Integrity & Exclusion Checks
    if str(row.get("blacklist_status", "")).strip().lower() == "clear":
        w = get_feature_weight("blacklist_status", fi_df)
        positive.append({
            "factor": "Entity clear of blacklisting records",
            "feature": "blacklist_status",
            "value": "Clear",
            "importance": w,
            "description": "No adverse debarment or blacklisting orders found across government portals."
        })

    if str(row.get("debarment_status", "")).strip().lower() == "not debarred":
        w = get_feature_weight("debarment_status", fi_df)
        positive.append({
            "factor": "No departmental debarment on record",
            "feature": "debarment_status",
            "value": "Not Debarred",
            "importance": w,
            "description": "Bidder is legally eligible to participate in central procurement tenders."
        })

    if str(row.get("tender_eligibility", "")).strip().lower() == "eligible":
        w = get_feature_weight("tender_eligibility", fi_df)
        positive.append({
            "factor": "Tender eligibility criteria satisfied",
            "feature": "tender_eligibility",
            "value": "Eligible",
            "importance": w,
            "description": "Bidder fulfills preliminary technical and commercial eligibility prerequisites, which contributed positively to compliance confidence."
        })

    # 5. Numerical Metrics
    try:
        comp_score = float(row.get("compliance_score", 0.0))
        if comp_score >= 75.0:
            w = get_feature_weight("compliance_score", fi_df)
            positive.append({
                "factor": f"High compliance index score ({comp_score:.1f} / 100)",
                "feature": "compliance_score",
                "value": comp_score,
                "importance": w,
                "description": "Composite statutory compliance index indicates solid historical standing and contributed positively to compliance confidence."
            })
    except (ValueError, TypeError):
        pass

    try:
        doc_comp = float(row.get("document_completeness", 0.0))
        if doc_comp >= 90.0:
            w = get_feature_weight("document_completeness", fi_df)
            positive.append({
                "factor": f"High document completeness ({doc_comp:.1f}%)",
                "feature": "document_completeness",
                "value": doc_comp,
                "importance": w,
                "description": "All mandatory tender document submissions were provided, which contributed positively to compliance confidence."
            })
    except (ValueError, TypeError):
        pass

    try:
        doc_cons = float(row.get("documentation_consistency", 0.0))
        if doc_cons >= 75.0:
            w = get_feature_weight("documentation_consistency", fi_df)
            positive.append({
                "factor": f"Strong cross-document consistency ({doc_cons:.1f}%)",
                "feature": "documentation_consistency",
                "value": doc_cons,
                "importance": w,
                "description": "Consistent bidder names and statutory registration identifiers across documents contributed positively to compliance confidence."
            })
    except (ValueError, TypeError):
        pass

    try:
        delay = int(row.get("verification_delay_days", 999))
        if delay <= 5:
            w = get_feature_weight("verification_delay_days", fi_df)
            positive.append({
                "factor": f"Prompt verification turnaround ({delay} days)",
                "feature": "verification_delay_days",
                "value": delay,
                "importance": w,
                "description": "Minimal delay in completing portal verifications indicates smooth records and contributed positively to the model prediction."
            })
    except (ValueError, TypeError):
        pass

    try:
        prev_iss = int(row.get("previous_compliance_issues", 999))
        if prev_iss == 0:
            w = get_feature_weight("previous_compliance_issues", fi_df)
            positive.append({
                "factor": "Zero past compliance defaults",
                "feature": "previous_compliance_issues",
                "value": prev_iss,
                "importance": w,
                "description": "Clean track record with zero reported compliance or delivery defaults contributed positively to the model prediction."
            })
    except (ValueError, TypeError):
        pass

    try:
        port_iss = int(row.get("portal_verification_issues", 999))
        if port_iss == 0:
            w = get_feature_weight("portal_verification_issues", fi_df)
            positive.append({
                "factor": "Zero automated portal verification errors",
                "feature": "portal_verification_issues",
                "value": port_iss,
                "importance": w,
                "description": "No schema mismatches or automated portal fetch errors encountered, which contributed positively to the model prediction."
            })
    except (ValueError, TypeError):
        pass

    try:
        loc_content = float(row.get("local_content_percentage", 0.0))
        if loc_content >= 50.0:
            w = get_feature_weight("local_content_percentage", fi_df)
            positive.append({
                "factor": f"Class-I Local Supplier ({loc_content:.1f}% domestic content)",
                "feature": "local_content_percentage",
                "value": loc_content,
                "importance": w,
                "description": "Qualifies for domestic preference under Make in India procurement guidelines, which contributed positively to compliance confidence."
            })
    except (ValueError, TypeError):
        pass

    # Sort descending by importance
    positive.sort(key=lambda x: x["importance"], reverse=True)
    return positive


def get_top_risk_factors(
    bid_data: Union[Dict[str, Any], pd.Series],
    fi_df: Optional[pd.DataFrame] = None
) -> List[Dict[str, Any]]:
    """
    Identifies risk indicators genuinely present in the submitted bid.
    Returns factors sorted by model importance weight.
    Does not claim causal relationship; factors contributed to model risk estimation.
    """
    row = bid_data.to_dict() if isinstance(bid_data, pd.Series) else dict(bid_data)
    risks: List[Dict[str, Any]] = []

    # 1. Statutory Exclusions
    if str(row.get("blacklist_status", "")).strip().lower() == "blacklisted":
        w = max(get_feature_weight("blacklist_status", fi_df), 0.15)
        risks.append({
            "factor": "Blacklist status detected on portal",
            "feature": "blacklist_status",
            "value": "Blacklisted",
            "importance": w,
            "is_exclusion": True,
            "description": "Vendor is recorded as blacklisted; statutory procurement rules mandate immediate exclusion review."
        })

    if str(row.get("debarment_status", "")).strip().lower() == "debarred":
        w = max(get_feature_weight("debarment_status", fi_df), 0.15)
        risks.append({
            "factor": "Active debarment status detected",
            "feature": "debarment_status",
            "value": "Debarred",
            "importance": w,
            "is_exclusion": True,
            "description": "Active government department debarment order requires disqualification review."
        })

    if str(row.get("tender_eligibility", "")).strip().lower() == "not eligible":
        w = max(get_feature_weight("tender_eligibility", fi_df), 0.12)
        risks.append({
            "factor": "Tender eligibility evaluated as 'Not Eligible'",
            "feature": "tender_eligibility",
            "value": "Not Eligible",
            "importance": w,
            "is_exclusion": True,
            "description": "Bidder fails preliminary mandatory tender criteria; contributes heavily to predicted non-compliance."
        })
    elif str(row.get("tender_eligibility", "")).strip().lower() == "requires review":
        w = get_feature_weight("tender_eligibility", fi_df)
        risks.append({
            "factor": "Tender eligibility marked 'Requires Review'",
            "feature": "tender_eligibility",
            "value": "Requires Review",
            "importance": w,
            "is_exclusion": False,
            "description": "Preliminary eligibility could not be automatically confirmed; officer scrutiny needed."
        })

    # 2. Statutory Registrations
    if str(row.get("gst_registration", "")).strip().lower() == "not verified":
        w = get_feature_weight("gst_registration", fi_df)
        risks.append({
            "factor": "GST registration not verified",
            "feature": "gst_registration",
            "value": "Not Verified",
            "importance": w,
            "is_exclusion": False,
            "description": "GSTIN could not be validated on GST portal; contributed to elevated risk score."
        })

    if str(row.get("gst_return_filing", "")).strip().lower() in ["not filed", "pending"]:
        val = str(row.get("gst_return_filing", "")).strip()
        w = get_feature_weight("gst_return_filing", fi_df)
        risks.append({
            "factor": f"GST return filing is {val}",
            "feature": "gst_return_filing",
            "value": val,
            "importance": w,
            "is_exclusion": False,
            "description": f"GST returns are {val.lower()}; introduces statutory compliance risk."
        })

    if str(row.get("pan_verification", "")).strip().lower() == "not verified":
        w = get_feature_weight("pan_verification", fi_df)
        risks.append({
            "factor": "PAN verification failed or unverified",
            "feature": "pan_verification",
            "value": "Not Verified",
            "importance": w,
            "is_exclusion": False,
            "description": "Permanent Account Number record could not be confirmed on tax database."
        })

    if str(row.get("income_tax_compliance", "")).strip().lower() in ["non-compliant", "pending"]:
        val = str(row.get("income_tax_compliance", "")).strip()
        w = get_feature_weight("income_tax_compliance", fi_df)
        risks.append({
            "factor": f"Income Tax status is {val}",
            "feature": "income_tax_compliance",
            "value": val,
            "importance": w,
            "is_exclusion": False,
            "description": f"Direct tax compliance record indicates {val.lower()} standing."
        })

    # 3. Labor & Technical
    if str(row.get("epfo_compliance", "")).strip().lower() == "non-compliant":
        w = get_feature_weight("epfo_compliance", fi_df)
        risks.append({
            "factor": "EPFO employee provident fund default",
            "feature": "epfo_compliance",
            "value": "Non-Compliant",
            "importance": w,
            "is_exclusion": False,
            "description": "Statutory labor provident fund non-compliance detected on portal records."
        })

    if str(row.get("esic_compliance", "")).strip().lower() == "non-compliant":
        w = get_feature_weight("esic_compliance", fi_df)
        risks.append({
            "factor": "ESIC employee insurance default",
            "feature": "esic_compliance",
            "value": "Non-Compliant",
            "importance": w,
            "is_exclusion": False,
            "description": "Statutory employee state insurance contributions flagged as non-compliant."
        })

    if str(row.get("oem_authorization", "")).strip().lower() == "invalid":
        w = get_feature_weight("oem_authorization", fi_df)
        risks.append({
            "factor": "OEM manufacturer authorization invalid or missing",
            "feature": "oem_authorization",
            "value": "Invalid",
            "importance": w,
            "is_exclusion": False,
            "description": "Mandatory manufacturer authorization certificate was not verified as active."
        })

    # 4. Numerical Metrics
    try:
        comp_score = float(row.get("compliance_score", 100.0))
        if comp_score < 60.0:
            w = get_feature_weight("compliance_score", fi_df)
            risks.append({
                "factor": f"Low compliance index score ({comp_score:.1f} / 100)",
                "feature": "compliance_score",
                "value": comp_score,
                "importance": w,
                "is_exclusion": False,
                "description": "Composite statutory compliance index is below acceptable operating threshold."
            })
    except (ValueError, TypeError):
        pass

    try:
        doc_comp = float(row.get("document_completeness", 100.0))
        if doc_comp < 85.0:
            w = get_feature_weight("document_completeness", fi_df)
            risks.append({
                "factor": f"Documentation gaps (Completeness: {doc_comp:.1f}%)",
                "feature": "document_completeness",
                "value": doc_comp,
                "importance": w,
                "is_exclusion": False,
                "description": "One or more mandatory tender documents were omitted from submission."
            })
    except (ValueError, TypeError):
        pass

    try:
        doc_cons = float(row.get("documentation_consistency", 100.0))
        if doc_cons < 70.0:
            w = get_feature_weight("documentation_consistency", fi_df)
            risks.append({
                "factor": f"Low documentation consistency ({doc_cons:.1f}%)",
                "feature": "documentation_consistency",
                "value": doc_cons,
                "importance": w,
                "is_exclusion": False,
                "description": "Cross-record inconsistencies detected between submitted documents and statutory databases."
            })
    except (ValueError, TypeError):
        pass

    try:
        delay = int(row.get("verification_delay_days", 0))
        if delay > 10:
            w = get_feature_weight("verification_delay_days", fi_df)
            risks.append({
                "factor": f"Elevated verification turnaround delay ({delay} days)",
                "feature": "verification_delay_days",
                "value": delay,
                "importance": w,
                "is_exclusion": False,
                "description": "Prolonged portal verification latency often associates with unconfirmed filings."
            })
    except (ValueError, TypeError):
        pass

    try:
        prev_iss = int(row.get("previous_compliance_issues", 0))
        if prev_iss >= 1:
            w = get_feature_weight("previous_compliance_issues", fi_df)
            risks.append({
                "factor": f"Prior compliance defaults on record ({prev_iss} incidents)",
                "feature": "previous_compliance_issues",
                "value": prev_iss,
                "importance": w,
                "is_exclusion": False,
                "description": "Vendor has a history of prior contract defaults or compliance inquiries on GeM."
            })
    except (ValueError, TypeError):
        pass

    try:
        port_iss = int(row.get("portal_verification_issues", 0))
        if port_iss >= 1:
            w = get_feature_weight("portal_verification_issues", fi_df)
            risks.append({
                "factor": f"Portal verification discrepancies ({port_iss} issues)",
                "feature": "portal_verification_issues",
                "value": port_iss,
                "importance": w,
                "is_exclusion": False,
                "description": "Automated data fetch produced schema mismatches or unconfirmed portal records."
            })
    except (ValueError, TypeError):
        pass

    try:
        loc_content = float(row.get("local_content_percentage", 100.0))
        if loc_content < 20.0:
            w = get_feature_weight("local_content_percentage", fi_df)
            risks.append({
                "factor": f"Domestic content below statutory threshold ({loc_content:.1f}%)",
                "feature": "local_content_percentage",
                "value": loc_content,
                "importance": w,
                "is_exclusion": False,
                "description": "Non-local supplier standing under Public Procurement Make in India clause."
            })
    except (ValueError, TypeError):
        pass

    # Sort descending by importance
    risks.sort(key=lambda x: x["importance"], reverse=True)
    return risks


def get_review_priority(
    assessment: Dict[str, Any],
    risk_factors: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Assigns prototype review priority based on risk level, statutory exclusion flags,
    and rule-based failed checks.
    
    Priority Tiers:
    - STANDARD REVIEW: Low Risk, minimal or no warnings.
    - PRIORITY REVIEW: Medium Risk, documentation gaps, moderate delays.
    - HIGH PRIORITY REVIEW: High Risk, low probabilities, critical failures.
    - Special Flag: STATUTORY EXCLUSION INDICATOR — OFFICER VERIFICATION REQUIRED
    """
    risk_level = assessment.get("risk_assessment", {}).get("risk_level", "Medium")
    pred_class = assessment.get("prediction", {}).get("predicted_compliance", 1)
    failed_count = assessment.get("verification", {}).get("total_failed", 0)

    has_statutory_exclusion = any(r.get("is_exclusion", False) for r in risk_factors)
    if assessment.get("procurement_review", {}).get("statutory_exclusions_flagged", False):
        has_statutory_exclusion = True

    if has_statutory_exclusion:
        priority_label = "HIGH PRIORITY REVIEW"
        badge_text = "STATUTORY EXCLUSION INDICATOR — OFFICER VERIFICATION REQUIRED"
        badge_color = "#c53030"
        urgency = "Immediate Officer Action Required"
        rationale = "Statutory disqualification grounds (blacklisting, debarment, or ineligibility) flagged on record."
    elif risk_level == "High" or pred_class == 0 or failed_count >= 2:
        priority_label = "HIGH PRIORITY REVIEW"
        badge_text = "HIGH PRIORITY REVIEW"
        badge_color = "#c53030"
        urgency = "High Urgency"
        rationale = "High compliance risk predicted by ML model with multiple statutory check defaults."
    elif risk_level == "Medium" or failed_count >= 1:
        priority_label = "PRIORITY REVIEW"
        badge_text = "PRIORITY REVIEW"
        badge_color = "#d69e2e"
        urgency = "Moderate Urgency"
        rationale = "Moderate compliance probability or documentation inconsistencies requiring Officer clarification."
    else:
        priority_label = "STANDARD REVIEW"
        badge_text = "STANDARD REVIEW"
        badge_color = "#2f855a"
        urgency = "Routine"
        rationale = "Standard routine verification; statutory checks and model predictions are satisfactory."

    return {
        "priority_label": priority_label,
        "badge_text": badge_text,
        "badge_color": badge_color,
        "urgency": urgency,
        "rationale": rationale,
        "statutory_exclusion": has_statutory_exclusion
    }


def generate_officer_summary(
    assessment: Dict[str, Any],
    positive_factors: List[Dict[str, Any]],
    risk_factors: List[Dict[str, Any]],
    review_priority: Dict[str, Any],
    doc_payload: Optional[Dict[str, Any]] = None
) -> str:
    """
    Constructs a concise, non-technical plain-language summary for the Procurement Officer.
    Adheres strictly to assistive language without asserting legal conclusions or causation.
    """
    risk_level = assessment.get("risk_assessment", {}).get("risk_level", "Medium")
    pred_class = assessment.get("prediction", {}).get("predicted_compliance", 1)
    comp_pct = assessment.get("prediction", {}).get("compliance_percentage", "N/A")
    non_comp_pct = assessment.get("prediction", {}).get("non_compliance_percentage", "N/A")
    company = assessment.get("company_name", "The bidder")

    top_risks_text = ", ".join([r["factor"].lower() for r in risk_factors[:3]]) if risk_factors else "no major risk indicators"
    top_pos_text = ", ".join([p["factor"].lower() for p in positive_factors[:3]]) if positive_factors else "standard criteria"

    if review_priority.get("statutory_exclusion", False):
        summary = (
            f"AI assessment indicates a high-risk submission for {company} requiring immediate Procurement Officer review. "
            f"Statutory exclusion indicators were detected ({top_risks_text}). "
            f"Model estimated non-compliance confidence at {non_comp_pct}. "
            "Final disqualification or qualification authority remains exclusively with the authorized Procurement Officer."
        )
    elif risk_level == "High" or pred_class == 0:
        summary = (
            f"AI assessment indicates a high-risk bid requiring Procurement Officer review. "
            f"The primary review indicators contributing to this assessment are {top_risks_text}. "
            f"The model estimated non-compliance probability at {non_comp_pct}. "
            "Verification of these flagged discrepancies with original documents is strongly advised."
        )
    elif risk_level == "Medium":
        summary = (
            f"AI assessment indicates a moderate-risk bid for {company} requiring Procurement Officer attention. "
            f"While {top_pos_text} contributed positively, notable review indicators include {top_risks_text}. "
            f"Compliance probability is estimated at {comp_pct}."
        )
    else:
        summary = (
            f"AI assessment indicates a lower-risk bid for {company} based on the available structured information. "
            f"Positive compliance indicators ({top_pos_text}) supported this assessment. "
            f"No critical prototype rule failures were detected (Compliance confidence: {comp_pct}). "
            "Standard officer verification is still required before tender award."
        )

    if doc_payload:
        doc_comp = doc_payload.get("document_completeness", "N/A")
        doc_cons = doc_payload.get("consistency_status", "N/A")
        summary += (
            f" Document verification module contributed additional context: "
            f"Document completeness is {doc_comp}%, and cross-document consistency is evaluated as {doc_cons}."
        )

    return summary


def generate_review_recommendations(
    assessment: Dict[str, Any],
    risk_factors: List[Dict[str, Any]],
    doc_payload: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Builds actionable, context-aware checklist items for Procurement Officer review.
    """
    recommendations: List[str] = []

    for r in risk_factors:
        feat = r.get("feature", "")
        if feat == "blacklist_status":
            recommendations.append("Mandatory Statutory Check: Verify active central and state blacklisting gazette notifications.")
        elif feat == "debarment_status":
            recommendations.append("Mandatory Statutory Check: Confirm departmental debarment order dates and applicability scope.")
        elif feat == "tender_eligibility":
            recommendations.append("Technical Evaluation: Re-examine preliminary eligibility proofs against tender notice clauses.")
        elif feat == "gst_registration":
            recommendations.append("Tax Verification: Perform live lookup of GSTIN on the GST portal (gst.gov.in) to verify active filing status.")
        elif feat == "pan_verification":
            recommendations.append("Identity Verification: Request fresh copy of PAN card and verify via Income Tax PAN verification service.")
        elif feat == "oem_authorization":
            recommendations.append("Technical Verification: Contact OEM manufacturer directly to authenticate authorization certificate validity.")
        elif feat == "epfo_compliance":
            recommendations.append("Labor Mandate: Verify Electronic Challan cum Return (ECR) statement on EPFO Unified Portal.")
        elif feat == "esic_compliance":
            recommendations.append("Labor Mandate: Verify monthly contribution challan receipt on ESIC portal.")
        elif feat == "document_completeness":
            recommendations.append("Documentation Audit: Request bidder submit missing mandatory tender documents within clarification window.")
        elif feat == "documentation_consistency":
            recommendations.append("Documentation Audit: Clarify legal entity name variations across statutory filings.")
        elif feat == "previous_compliance_issues":
            recommendations.append("Past Performance: Review incident reports regarding previous GeM contractual performance defaults.")

    if doc_payload and doc_payload.get("document_completeness", 100.0) < 100.0:
        recommendations.append(
            f"Document Audit: Address missing document types flagged during automated text extraction (Completeness: {doc_payload.get('document_completeness')}%)."
        )

    if not recommendations:
        recommendations.append("Proceed with standard procedural tender scrutiny and physical document cross-check.")
        recommendations.append("Confirm statutory exemption claims (if any) against original certificates.")

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for rec in recommendations:
        if rec not in seen:
            seen.add(rec)
            deduped.append(rec)

    return deduped


def explain_prediction(
    bid_data: Union[Dict[str, Any], pd.DataFrame, pd.Series],
    model=None,
    feature_importance_df: Optional[pd.DataFrame] = None,
    assessment: Optional[Dict[str, Any]] = None,
    doc_payload: Optional[Dict[str, Any]] = None,
    doc_verification_payload: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Main explainability orchestrator for AI-assisted compliance intelligence.
    
    Combines:
    1. Pre-computed or real-time assess_bid output
    2. Model feature signals & directional contributions
    3. Positive compliance indicators vs. risk indicators
    4. Categorized rule-based verification findings (Passed, Warnings, Failed)
    5. Review priority tiering
    6. Non-technical Procurement Officer summary and review recommendations
    7. Integration with document verification results
    """
    row = bid_data.iloc[0].to_dict() if isinstance(bid_data, pd.DataFrame) else (
        bid_data.to_dict() if isinstance(bid_data, pd.Series) else dict(bid_data)
    )

    effective_doc_payload = doc_payload if doc_payload is not None else doc_verification_payload

    if feature_importance_df is not None and not feature_importance_df.empty:
        fi_table = feature_importance_df
    else:
        fi_table = load_feature_importance_table()

    # Step 1: Obtain or execute assess_bid
    if assessment is None:
        if model is None:
            model = load_model()
        assessment = assess_bid(row, model=model)

    # Step 2: Extract positive and risk factors
    positive_factors = get_top_positive_factors(row, fi_table)
    risk_factors = get_top_risk_factors(row, fi_table)

    # Step 3: Compute Review Priority
    priority_info = get_review_priority(assessment, risk_factors)

    # Step 4: Construct Procurement Officer Summary
    officer_summary = generate_officer_summary(
        assessment, positive_factors, risk_factors, priority_info, doc_payload=effective_doc_payload
    )

    # Step 5: Generate Review Recommendations
    recommendations = generate_review_recommendations(
        assessment, risk_factors, doc_payload=effective_doc_payload
    )

    # Step 6: Extract Top Model Signals for visualization
    # Build list of feature contributions for chart
    model_signals: List[Dict[str, Any]] = []
    for rf in risk_factors[:8]:
        model_signals.append({
            "feature_label": rf["factor"],
            "importance": round(rf["importance"], 4),
            "signal_type": "Risk Signal",
            "direction": "Elevated Risk",
            "color": "#e76f51"
        })
    for pf in positive_factors[:8]:
        model_signals.append({
            "feature_label": pf["factor"],
            "importance": round(pf["importance"], 4),
            "signal_type": "Positive Signal",
            "direction": "Reduced Risk",
            "color": "#2a9d8f"
        })

    model_signals.sort(key=lambda x: x["importance"], reverse=True)

    return {
        "bid_id": assessment.get("bid_id", row.get("bid_id", "BID_INPUT")),
        "company_name": assessment.get("company_name", row.get("company_name", "Bidder")),
        "prediction_outcome": assessment.get("prediction", {}).get("predicted_compliance", 1),
        "prediction_label": "Compliant" if assessment.get("prediction", {}).get("predicted_compliance", 1) == 1 else "Non-Compliant / Review",
        "compliance_probability": assessment.get("prediction", {}).get("compliance_probability", 0.5),
        "non_compliance_probability": assessment.get("prediction", {}).get("non_compliance_probability", 0.5),
        "compliance_percentage": assessment.get("prediction", {}).get("compliance_percentage", "50.0%"),
        "non_compliance_percentage": assessment.get("prediction", {}).get("non_compliance_percentage", "50.0%"),
        "risk_level": assessment.get("risk_assessment", {}).get("risk_level", "Medium"),
        "review_priority": priority_info["priority_label"],
        "review_badge": priority_info["badge_text"],
        "review_badge_color": priority_info["badge_color"],
        "urgency": priority_info["urgency"],
        "priority_rationale": priority_info["rationale"],
        "positive_factors": positive_factors,
        "risk_factors": risk_factors,
        "top_positive_indicators": [p["factor"] for p in positive_factors[:5]],
        "top_risk_indicators": [r["factor"] for r in risk_factors[:5]],
        "rule_based_checks": {
            "passed": assessment.get("verification", {}).get("passed_checks", []),
            "warnings": assessment.get("verification", {}).get("warning_checks", []),
            "failed": assessment.get("verification", {}).get("failed_checks", [])
        },
        "officer_summary": officer_summary,
        "recommendations": recommendations,
        "model_signals": model_signals[:15],
        "doc_verification_context": {
            "available": doc_payload is not None,
            "document_completeness": doc_payload.get("document_completeness") if doc_payload else None,
            "documentation_consistency": doc_payload.get("documentation_consistency") if doc_payload else None,
            "consistency_status": doc_payload.get("consistency_status") if doc_payload else None,
            "explanation": (
                f"Document analysis contributed additional review indicators: "
                f"Completeness = {doc_payload.get('document_completeness')}%, "
                f"Consistency = {doc_payload.get('documentation_consistency')}% ({doc_payload.get('consistency_status')})."
            ) if doc_payload else None
        },
        "disclaimer": EXPLAINABILITY_DISCLAIMER
    }
