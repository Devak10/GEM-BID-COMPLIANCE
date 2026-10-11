"""
AI-Powered Integrated Bid Compliance Verification Platform for GeM Procurement
AIML Hackathon Problem Statement ID: 26100

Module: Dataset Generation
File: src/create_dataset.py

DISCLAIMER:
"This dataset is synthetic/demo data created solely for AIML prototype development and testing.
It is not official Government of India, GeM, CPCL, GSTN, Udyam, MCA, or any other government data."
No real personal data or confidential government records are used or claimed to be used.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd


def generate_bid_compliance_dataset(
    num_records: int = 1000,
    random_seed: int = 42,
    output_path: str = "data/bid_compliance_dataset.csv"
) -> pd.DataFrame:
    """
    Generates a reproducible, logically consistent synthetic dataset representing
    bidder compliance evaluations for GeM procurement tenders.
    """
    np.random.seed(random_seed)

    # 1. State and District mapping (Indian states & realistic districts)
    state_district_map = {
        "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem"],
        "Kerala": ["Thiruvananthapuram", "Ernakulam", "Kozhikode", "Thrissur", "Kollam"],
        "Karnataka": ["Bengaluru Urban", "Mysuru", "Hubballi-Dharwad", "Mangaluru", "Belagavi"],
        "Andhra Pradesh": ["Visakhapatnam", "Vijayawada", "Guntur", "Tirupati", "Kurnool"],
        "Telangana": ["Hyderabad", "Warangal", "Medchal-Malkajgiri", "Rangareddy", "Nizamabad"],
        "Maharashtra": ["Mumbai City", "Pune", "Nagpur", "Thane", "Nashik"],
        "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar"],
        "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota", "Ajmer"],
        "Delhi": ["New Delhi", "North Delhi", "South Delhi", "West Delhi", "East Delhi"],
        "Uttar Pradesh": ["Lucknow", "Kanpur", "Varanasi", "Noida", "Prayagraj"],
        "West Bengal": ["Kolkata", "Howrah", "North 24 Parganas", "Darjeeling", "Siliguri"],
        "Odisha": ["Khordha (Bhubaneswar)", "Cuttack", "Sundargarh", "Ganjam", "Sambalpur"]
    }
    states = list(state_district_map.keys())

    # 2. Project Types
    project_types = [
        "Infrastructure",
        "Construction",
        "Electrical",
        "Mechanical",
        "IT Services",
        "Facility Management",
        "Transportation",
        "Consultancy",
        "Equipment Supply",
        "Maintenance"
    ]

    # 3. Synthetic Company Name Components
    company_prefixes = [
        "Apex", "Bharat", "Chola", "Coromandel", "Deccan", "Eastern", "Hind",
        "Imperial", "Kaveri", "Maruti", "National", "Pacific", "Prime", "Quantum",
        "Reliable", "SmartTech", "Sunrise", "Titan", "Vanguard", "Western", "Zenith",
        "ABC", "XYZ", "Alpha", "Delta", "Metro", "United", "Surya", "Garuda"
    ]
    company_cores = [
        "Infrastructure", "Engineering Solutions", "BuildTech Services",
        "Industrial Systems", "Procurement Solutions", "Technologies", "Power Systems",
        "Electricals", "Logistics & Transport", "Heavy Engineering", "Facility Management",
        "Technical Services", "Equipments & Tools", "Project Consultants", "IT Solutions",
        "Civil Works", "Data Systems", "Fabrication Works", "Electro-Mech Systems"
    ]
    company_suffixes = ["Pvt Ltd", "Ltd", "LLP", "Enterprises", "Corporation", "Solutions"]

    # Pre-generate a pool of 350 distinct synthetic bidders to ensure realistic recurring participation
    num_bidders = 350
    bidder_pool = []
    for i in range(1, num_bidders + 1):
        b_id = f"BIDDER{i:04d}"
        pref = company_prefixes[(i * 7) % len(company_prefixes)]
        core = company_cores[(i * 13) % len(company_cores)]
        suff = company_suffixes[(i * 3) % len(company_suffixes)]
        comp_name = f"{pref} {core} {suff}"
        st = states[i % len(states)]
        dist = state_district_map[st][i % len(state_district_map[st])]
        bidder_pool.append({
            "bidder_id": b_id,
            "company_name": comp_name,
            "state": st,
            "district": dist,
            "base_reputation": np.random.uniform(0.3, 0.98)
        })

    # Pre-generate a pool of 180 distinct tenders
    num_tenders = 180
    tender_pool = [f"TND{i:04d}" for i in range(1, num_tenders + 1)]

    # Containers for generated rows
    records = []

    for i in range(1, num_records + 1):
        bid_id = f"BID{i:04d}"
        tender_id = np.random.choice(tender_pool)
        bidder = np.random.choice(bidder_pool)

        bidder_id = bidder["bidder_id"]
        company_name = bidder["company_name"]
        state = bidder["state"]
        district = bidder["district"]
        rep = bidder["base_reputation"]

        project_type = np.random.choice(project_types)

        # Bid value in Lakhs (5 Lakhs to 2500 Lakhs)
        # Lognormal distribution to simulate realistic skew toward small-to-mid value bids
        raw_val = np.random.lognormal(mean=4.2, sigma=1.1)
        bid_value_lakhs = round(float(np.clip(raw_val, 5.0, 2500.0)), 2)

        # Document counts
        req_docs = int(np.random.choice([6, 7, 8, 9, 10, 11, 12, 14, 15]))
        
        # Submitted documents depends on reputation
        if rep > 0.75:
            # High reputation vendors submit all or almost all required documents
            sub_docs = req_docs if np.random.rand() > 0.15 else req_docs - 1
        elif rep > 0.5:
            sub_docs = req_docs - int(np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1]))
        else:
            missing = int(np.random.choice([1, 2, 3, 4], p=[0.4, 0.3, 0.2, 0.1]))
            sub_docs = max(2, req_docs - missing)

        sub_docs = min(sub_docs, req_docs)
        doc_completeness = round(float((sub_docs / req_docs) * 100.0), 2)

        # Documentation consistency percentage (0 to 100)
        base_cons = rep * 95 + np.random.normal(0, 8)
        doc_consistency = round(float(np.clip(base_cons, 20.0, 100.0)), 2)

        # Blacklist and Debarment status (low probability overall)
        # Higher probability only for very low reputation entities
        is_blacklisted = (rep < 0.38 and np.random.rand() < 0.25) or (np.random.rand() < 0.015)
        blacklist_status = "Blacklisted" if is_blacklisted else "Clear"

        is_debarred = (rep < 0.42 and np.random.rand() < 0.22) or (np.random.rand() < 0.018)
        debarment_status = "Debarred" if is_debarred else "Not Debarred"

        # GST & PAN Verification
        if is_blacklisted or is_debarred:
            gst_registration = np.random.choice(["Verified", "Not Verified"], p=[0.3, 0.7])
            gst_return_filing = np.random.choice(["Filed", "Pending", "Not Filed"], p=[0.2, 0.3, 0.5])
            pan_verification = np.random.choice(["Verified", "Not Verified"], p=[0.5, 0.5])
            it_compliance = np.random.choice(["Compliant", "Pending", "Non-Compliant"], p=[0.1, 0.3, 0.6])
        else:
            p_gst = 0.96 if rep > 0.6 else 0.75
            gst_registration = "Verified" if np.random.rand() < p_gst else "Not Verified"

            if gst_registration == "Verified":
                gst_return_filing = np.random.choice(["Filed", "Pending", "Not Filed"], p=[0.88, 0.09, 0.03])
            else:
                gst_return_filing = np.random.choice(["Filed", "Pending", "Not Filed"], p=[0.25, 0.35, 0.40])

            p_pan = 0.98 if rep > 0.5 else 0.85
            pan_verification = "Verified" if np.random.rand() < p_pan else "Not Verified"

            it_compliance = np.random.choice(["Compliant", "Pending", "Non-Compliant"], p=[0.82, 0.12, 0.06])

        # Udyam MSME Registration
        # MSME might be Not Applicable for large corporations
        is_msme = np.random.rand() < 0.72
        if is_msme:
            udyam_reg = "Verified" if (rep > 0.45 and np.random.rand() > 0.08) else "Not Verified"
        else:
            udyam_reg = "Not Applicable"

        # Startup India & NSIC Status
        if udyam_reg == "Verified" and np.random.rand() < 0.30:
            startup_status = "Verified" if np.random.rand() > 0.15 else "Not Verified"
        else:
            startup_status = "Not Applicable"

        if udyam_reg == "Verified" and np.random.rand() < 0.25:
            nsic_status = "Verified" if np.random.rand() > 0.10 else "Not Verified"
        else:
            nsic_status = "Not Applicable"

        # EPFO & ESIC Compliance (Applicable based on company scale / project type)
        # Infrastructure, Construction, Facility Management, Transportation usually require labor statutory compliance
        labor_intensive = project_type in [
            "Infrastructure", "Construction", "Facility Management",
            "Transportation", "Maintenance", "Electrical", "Mechanical"
        ]
        if labor_intensive or np.random.rand() < 0.80:
            epfo_status = "Compliant" if (rep > 0.48 and np.random.rand() > 0.10) else "Non-Compliant"
            esic_status = "Compliant" if (rep > 0.48 and np.random.rand() > 0.10) else "Non-Compliant"
        else:
            epfo_status = "Not Applicable"
            esic_status = "Not Applicable"

        # OEM Authorization
        # Typically applicable for Equipment Supply, Mechanical, Electrical, IT Services
        oem_relevant = project_type in ["Equipment Supply", "Mechanical", "Electrical", "IT Services"]
        if oem_relevant:
            oem_auth = "Valid" if (rep > 0.5 and np.random.rand() > 0.12) else "Invalid"
        else:
            oem_auth = "Not Applicable"

        # Local Content Percentage & Make in India Compliance
        # For domestic public procurement, standard threshold is 50% for Class-I and 20% for Class-II
        if rep > 0.6:
            local_content = round(float(np.clip(np.random.normal(72, 15), 35.0, 100.0)), 2)
        else:
            local_content = round(float(np.clip(np.random.normal(48, 22), 0.0, 95.0)), 2)

        # Make in India compliance alignment
        if project_type in ["Consultancy", "Facility Management"]:
            # Services might occasionally treat it as Not Applicable or Compliant
            mii_compliance = "Compliant" if local_content >= 50.0 else "Non-Compliant"
        else:
            if local_content >= 50.0:
                mii_compliance = "Compliant" if np.random.rand() > 0.05 else "Non-Compliant"
            elif local_content >= 20.0:
                mii_compliance = "Compliant" if np.random.rand() > 0.35 else "Non-Compliant"
            else:
                mii_compliance = "Non-Compliant"

        # DigiLocker Verification
        digilocker = np.random.choice(["Verified", "Not Verified", "Not Applicable"], p=[0.70, 0.20, 0.10])

        # Previous compliance issues (0 to 5)
        if rep > 0.8:
            prev_issues = int(np.random.choice([0, 1], p=[0.90, 0.10]))
        elif rep > 0.5:
            prev_issues = int(np.random.choice([0, 1, 2], p=[0.65, 0.25, 0.10]))
        else:
            prev_issues = int(np.random.choice([1, 2, 3, 4, 5], p=[0.30, 0.30, 0.20, 0.15, 0.05]))

        # Portal verification issues (0 to 4)
        if gst_registration == "Verified" and pan_verification == "Verified" and rep > 0.6:
            portal_issues = int(np.random.choice([0, 1], p=[0.85, 0.15]))
        else:
            portal_issues = int(np.random.choice([0, 1, 2, 3, 4], p=[0.25, 0.35, 0.20, 0.15, 0.05]))

        # Verification delay in days (0 to 30)
        delay_mu = 3.0 if rep > 0.7 else (8.0 if rep > 0.45 else 16.0)
        delay_days = int(np.clip(np.random.poisson(lam=delay_mu) + prev_issues * 2 + portal_issues * 2, 0, 30))

        # Tender Eligibility:
        # Eligible, Requires Review, Not Eligible
        major_red_flag = (
            is_blacklisted
            or is_debarred
            or gst_registration == "Not Verified"
            or pan_verification == "Not Verified"
            or (oem_relevant and oem_auth == "Invalid")
        )
        if major_red_flag:
            tender_eligibility = np.random.choice(["Not Eligible", "Requires Review"], p=[0.82, 0.18])
        elif doc_completeness < 75.0 or doc_consistency < 60.0 or it_compliance == "Non-Compliant" or portal_issues >= 2:
            tender_eligibility = np.random.choice(["Requires Review", "Eligible"], p=[0.70, 0.30])
        else:
            tender_eligibility = "Eligible" if np.random.rand() > 0.08 else "Requires Review"

        # ==================================================
        # COMPLIANCE SCORE CALCULATION (0 to 100)
        # ==================================================
        # Factor 1: GST (15 pts)
        score_gst = (8.0 if gst_registration == "Verified" else 0.0) + (
            7.0 if gst_return_filing == "Filed" else (3.5 if gst_return_filing == "Pending" else 0.0)
        )

        # Factor 2: PAN & IT (12 pts)
        score_pan_it = (6.0 if pan_verification == "Verified" else 0.0) + (
            6.0 if it_compliance == "Compliant" else (3.0 if it_compliance == "Pending" else 0.0)
        )

        # Factor 3: Statutory & Social Security (15 pts)
        score_statutory = (
            (5.0 if udyam_reg == "Verified" else (4.0 if udyam_reg == "Not Applicable" else 0.0))
            + (5.0 if epfo_status == "Compliant" else (4.0 if epfo_status == "Not Applicable" else 0.0))
            + (5.0 if esic_status == "Compliant" else (4.0 if esic_status == "Not Applicable" else 0.0))
        )

        # Factor 4: OEM & MII & DigiLocker (15 pts)
        score_policy = (
            (5.0 if oem_auth == "Valid" else (4.0 if oem_auth == "Not Applicable" else 0.0))
            + (5.0 if mii_compliance == "Compliant" else (3.5 if mii_compliance == "Not Applicable" else 0.0))
            + (5.0 if digilocker == "Verified" else (3.5 if digilocker == "Not Applicable" else 1.0))
        )

        # Factor 5: Document Completeness (15 pts) & Consistency (10 pts)
        score_docs = (doc_completeness / 100.0) * 15.0 + (doc_consistency / 100.0) * 10.0

        # Factor 6: Tender Eligibility (8 pts)
        score_elig = 8.0 if tender_eligibility == "Eligible" else (4.0 if tender_eligibility == "Requires Review" else 0.0)

        # Factor 7: Local Content (5 pts)
        score_local = (local_content / 100.0) * 5.0

        # Factor 8: Startup / NSIC recognition (up to 5 pts bonus)
        score_bonus = 0.0
        if startup_status == "Verified":
            score_bonus += 2.5
        if nsic_status == "Verified":
            score_bonus += 2.5

        raw_score = (
            score_gst
            + score_pan_it
            + score_statutory
            + score_policy
            + score_docs
            + score_elig
            + score_local
            + score_bonus
        )

        # Penalties:
        penalty = 0.0
        if is_blacklisted:
            penalty += 45.0
        if is_debarred:
            penalty += 40.0
        penalty += min(12.0, prev_issues * 3.0)
        penalty += min(9.0, portal_issues * 2.5)
        penalty += min(10.0, delay_days * 0.4)

        # Add minor natural stochastic variation (+- 1.5 pts)
        noise = np.random.normal(0, 1.2)
        final_score = raw_score - penalty + noise
        compliance_score = round(float(np.clip(final_score, 0.0, 100.0)), 2)

        # ==================================================
        # RISK LEVEL DETERMINATION
        # Allowed values: Low, Medium, High
        # ==================================================
        if (
            is_blacklisted
            or is_debarred
            or gst_registration == "Not Verified"
            or pan_verification == "Not Verified"
            or tender_eligibility == "Not Eligible"
            or compliance_score < 48.0
            or prev_issues >= 3
        ):
            risk_level = "High"
        elif (
            compliance_score < 74.0
            or tender_eligibility == "Requires Review"
            or doc_completeness < 80.0
            or doc_consistency < 70.0
            or gst_return_filing != "Filed"
            or it_compliance != "Compliant"
            or portal_issues >= 2
            or (oem_relevant and oem_auth == "Invalid")
        ):
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # ==================================================
        # TARGET VARIABLE: actual_compliance (0 or 1)
        # Probabilistic multi-factor relationship without direct target leakage
        # ==================================================
        if is_blacklisted or is_debarred:
            # Disqualified by statutory mandate
            actual_compliance = 0
        elif gst_registration == "Not Verified" or pan_verification == "Not Verified":
            # Fundamental identity failure: negligible probability of compliance
            actual_compliance = 1 if np.random.rand() < 0.03 else 0
        else:
            # Latent logit combination of multiple underlying features
            # Centered around baseline threshold
            logit = (
                0.045 * (compliance_score - 65.0)
                + 0.025 * (doc_completeness - 75.0)
                + 0.020 * (doc_consistency - 70.0)
                + (0.8 if gst_return_filing == "Filed" else (-0.7 if gst_return_filing == "Not Filed" else 0.0))
                + (0.7 if it_compliance == "Compliant" else (-0.8 if it_compliance == "Non-Compliant" else -0.2))
                + (0.9 if tender_eligibility == "Eligible" else (-1.1 if tender_eligibility == "Not Eligible" else -0.3))
                + (-0.4 * prev_issues)
                + (-0.35 * portal_issues)
                + (-0.05 * delay_days)
                + (0.6 if mii_compliance == "Compliant" else -0.4)
                + (0.6 if (not oem_relevant or oem_auth == "Valid") else -1.2)
                + np.random.normal(0, 0.45)  # Stochastic unobserved factor (e.g. site inspection, physical sample)
            )
            prob_compliant = 1.0 / (1.0 + np.exp(-logit))
            actual_compliance = 1 if np.random.rand() < prob_compliant else 0

        # Compile record matching the exact 34 required columns
        record = {
            "bid_id": bid_id,
            "tender_id": tender_id,
            "bidder_id": bidder_id,
            "company_name": company_name,
            "project_type": project_type,
            "state": state,
            "district": district,
            "bid_value_lakhs": bid_value_lakhs,
            "gst_registration": gst_registration,
            "gst_return_filing": gst_return_filing,
            "pan_verification": pan_verification,
            "udyam_registration": udyam_reg,
            "income_tax_compliance": it_compliance,
            "startup_india_status": startup_status,
            "nsic_status": nsic_status,
            "epfo_compliance": epfo_status,
            "esic_compliance": esic_status,
            "oem_authorization": oem_auth,
            "make_in_india_compliance": mii_compliance,
            "local_content_percentage": local_content,
            "digilocker_verification": digilocker,
            "blacklist_status": blacklist_status,
            "debarment_status": debarment_status,
            "required_documents": req_docs,
            "submitted_documents": sub_docs,
            "document_completeness": doc_completeness,
            "documentation_consistency": doc_consistency,
            "tender_eligibility": tender_eligibility,
            "previous_compliance_issues": prev_issues,
            "portal_verification_issues": portal_issues,
            "verification_delay_days": delay_days,
            "compliance_score": compliance_score,
            "risk_level": risk_level,
            "actual_compliance": int(actual_compliance)
        }
        records.append(record)

    df = pd.DataFrame(records)

    # ==================================================
    # DATA VALIDATION
    # ==================================================
    expected_columns = [
        "bid_id", "tender_id", "bidder_id", "company_name", "project_type",
        "state", "district", "bid_value_lakhs", "gst_registration", "gst_return_filing",
        "pan_verification", "udyam_registration", "income_tax_compliance",
        "startup_india_status", "nsic_status", "epfo_compliance", "esic_compliance",
        "oem_authorization", "make_in_india_compliance", "local_content_percentage",
        "digilocker_verification", "blacklist_status", "debarment_status",
        "required_documents", "submitted_documents", "document_completeness",
        "documentation_consistency", "tender_eligibility", "previous_compliance_issues",
        "portal_verification_issues", "verification_delay_days", "compliance_score",
        "risk_level", "actual_compliance"
    ]

    # 1. Exactly 1000 rows
    assert len(df) == num_records, f"Validation Failed: Expected {num_records} rows, got {len(df)}"

    # 2. All required columns exist
    missing_cols = set(expected_columns) - set(df.columns)
    assert not missing_cols, f"Validation Failed: Missing columns: {missing_cols}"
    df = df[expected_columns]  # Ensure exact column order

    # 3. bid_id values are unique
    assert df["bid_id"].nunique() == num_records, "Validation Failed: Duplicate bid_id detected"

    # 4. tender_id valid
    assert df["tender_id"].notna().all() and (df["tender_id"] != "").all(), "Validation Failed: Invalid tender_id values"

    # 5. bidder_id valid
    assert df["bidder_id"].notna().all() and (df["bidder_id"] != "").all(), "Validation Failed: Invalid bidder_id values"

    # 6. actual_compliance contains only 0 and 1
    assert set(df["actual_compliance"].unique()).issubset({0, 1}), "Validation Failed: actual_compliance not in {0, 1}"

    # 7. compliance_score is between 0 and 100
    assert (df["compliance_score"] >= 0.0).all() and (df["compliance_score"] <= 100.0).all(), "Validation Failed: compliance_score out of range"

    # 8. local_content_percentage is between 0 and 100
    assert (df["local_content_percentage"] >= 0.0).all() and (df["local_content_percentage"] <= 100.0).all(), "Validation Failed: local_content_percentage out of range"

    # 9. document_completeness is between 0 and 100
    assert (df["document_completeness"] >= 0.0).all() and (df["document_completeness"] <= 100.0).all(), "Validation Failed: document_completeness out of range"

    # 10. documentation_consistency is between 0 and 100
    assert (df["documentation_consistency"] >= 0.0).all() and (df["documentation_consistency"] <= 100.0).all(), "Validation Failed: documentation_consistency out of range"

    # 11. submitted_documents does not exceed required_documents
    assert (df["submitted_documents"] <= df["required_documents"]).all(), "Validation Failed: submitted_documents exceeds required_documents"

    # 12. risk_level contains only Low, Medium, High
    assert set(df["risk_level"].unique()).issubset({"Low", "Medium", "High"}), "Validation Failed: Invalid risk_level categories"

    # 13. No required column is completely empty
    for col in expected_columns:
        assert df[col].notna().all(), f"Validation Failed: Column '{col}' contains NaN values"

    # Ensure output directory exists
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(out_file, index=False)

    return df


if __name__ == "__main__":
    csv_target = "data/bid_compliance_dataset.csv"
    generated_df = generate_bid_compliance_dataset(
        num_records=1000,
        random_seed=42,
        output_path=csv_target
    )

    print("Dataset created successfully.\n")
    print("Dataset shape:")
    print(generated_df.shape)
    print("\nTarget distribution:")
    print(generated_df["actual_compliance"].value_counts().to_string())
    print("\nRisk level distribution:")
    print(generated_df["risk_level"].value_counts().to_string())
    print("\nFirst 5 rows:")
    print(generated_df.head(5).to_string())
    print("\nCSV path:")
    print(csv_target)
    print("\nList of column names:")
    for idx, col in enumerate(generated_df.columns, 1):
        print(f"{idx}. {col}")
