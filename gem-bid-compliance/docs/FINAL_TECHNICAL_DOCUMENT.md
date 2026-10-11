# FINAL TECHNICAL SPECIFICATION & SYSTEM ARCHITECTURE DOCUMENT

**Project Title:** AI-Powered Integrated Bid Compliance Verification Platform for GeM Procurement  
**Assigned Problem Statement ID:** 26100  
**Organization:** Chennai Petroleum Corporation Limited (CPCL)  
**Theme:** Smart Automation  
**Hackathon Event:** AIML Hackathon  
**Target Repository:** `https://github.com/Devak10/GEM-BID-COMPLIANCE`  
**Current Status:** Prototype Complete & Hackathon Demo Ready  

---

## 1. Executive Summary

Public procurement operations governed by platforms like Government e-Marketplace (GeM) handle substantial volumes of commercial, technical, and statutory documentation submitted by prospective contractors. Evaluating bidder eligibility manually imposes a significant administrative burden on Procurement Officers, creating operational bottlenecks and introducing risks of oversight.

The **AI-Powered Integrated Bid Compliance Verification Platform** is an intelligent, assistive decision-support platform designed to automate preliminary compliance screening while maintaining strict **human-in-the-loop governance**. The platform integrates:
1. **Calibrated Machine Learning:** Predicting statistical compliance probabilities based on historical bid signals.
2. **Deterministic Statutory Rules:** Enforcing 21 explicit regulatory criteria (GST validity, PAN status, MSME preferences, local content thresholds, and central debarment/blacklisting registries).
3. **AI-Assisted Document Verification:** Extracting and validating text content from PDF and TXT submissions with cross-document consistency checks.
4. **Transparent Explainability:** Decomposing predictions into positive compliance factors and risk indicators using non-causal language.
5. **State-Machine Review Workflow:** Logging every officer action, comment, and lifecycle state change to an immutable SQLite audit trail.

---

## 2. Problem Statement

- **Problem Statement ID:** 26100
- **Organization:** Chennai Petroleum Corporation Limited (CPCL)
- **Theme:** Smart Automation
- **Challenge Overview:** In modern industrial procurement, bids must be scrutinized against multiple statutory, financial, and technical parameters. Procurement officers must verify diverse documentation (tax compliance certificates, MSME standing, OEM authorizations, statutory labor standing, and debarment registries) across dozens of competitors per tender. Manual scrutiny is repetitive, time-intensive, and susceptible to cognitive fatigue, while inconsistent evaluations may lead to disputes. The objective is to build an intelligent, integrated automation platform that pre-screens bidder compliance while preserving the officer's statutory decision authority.

---

## 3. Proposed Solution

The proposed platform provides an end-to-end evaluation environment structured into three distinct operational layers:
- **Layer 1: Predictive Machine Learning:** Leverages a trained Random Forest classifier to identify non-linear compliance patterns and estimate an overall compliance probability.
- **Layer 2: Deterministic Statutory Verification:** Evaluates 21 explicit statutory rules. Crucially, active statutory exclusion indicators (`Blacklisted`, `Debarred`, `Not Eligible`) instantly escalate the risk classification to `High` and mandate human scrutiny regardless of model confidence.
- **Layer 3: Human-in-the-Loop Governance & Audit Trail:** Provides a structured procurement officer review interface with immutable SQLite logging. The system assists but never usurps the procurement officer's authority.

---

## 4. Functional Requirements

1. **Structured Bid Ingestion:** Ingest structured bidder attributes across commercial, technical, and statutory domains.
2. **Automated Rule Screening:** Execute 21 deterministic rule checks evaluating GST, PAN, Udyam, labor compliance, OEM authorization, local content, and debarment status.
3. **ML Inference & Risk Classification:** Compute compliance probability and categorize risk into `Low`, `Medium`, and `High` screening tiers.
4. **Document Ingestion & Parsing:** Parse PDF and plain-text submissions, extract textual content, and identify document categories based on keyword profiles.
5. **Document Keyword Verification:** Inspect documents for mandatory statutory terminology and flag omissions.
6. **Cross-Document Entity Matching:** Perform sequence matching across document entity names and statutory IDs to identify cross-record discrepancies.
7. **Document Metric Transfer:** Seamlessly inject document completeness and consistency scores into active bid verification sessions.
8. **Explainable AI Generation:** Generate transparent positive indicators, risk indicators, and natural-language summaries explaining model predictions.
9. **Procurement Review Workflow:** Support review state transitions (`NOT REVIEWED` $\rightarrow$ `UNDER REVIEW` $\rightarrow$ `REVIEWED` or `ESCALATED`).
10. **Immutable Audit Logging:** Record all officer interactions, remarks, and status changes in a persistent SQLite database.
11. **Analytical Reporting:** Render interactive dashboard KPIs, geographic breakdowns, sector distributions, and model benchmarking metrics.

---

## 5. Non-Functional Requirements

1. **Performance & Latency:** Single-bid inference and rule evaluation execute in $< 50$ milliseconds; document text extraction and verification complete in $< 1.5$ seconds per file.
2. **Reliability & Crash Resilience:** Model inference and document extraction gracefully handle corrupted files, missing schema values, and unknown categorical variables using robust imputation and exception containment.
3. **Explainability & Trust:** Explanations strictly employ non-causal terminology (*"Feature contributed to prediction"* instead of *"Feature caused decision"*) to avoid deceptive certainty.
4. **Auditability & Traceability:** Audit records are append-only; historical logs cannot be mutated or purged through the application interface.
5. **Data Privacy & Security:** Operates locally without sending confidential bidder data to third-party public cloud APIs; database files are excluded from version control.

---

## 6. System Architecture

```text
+-----------------------------------------------------------------------------------+
|                            PRESENTATION TIER (Streamlit)                          |
|  [Dashboard] [Bid Verification] [Doc Verification] [Risk Analysis] [Officer Review]|
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                             BUSINESS & INFERENCE TIER                             |
|                                                                                   |
|  +-----------------------------+       +---------------------------------------+  |
|  |   Document Verification     |       |       Prediction Engine               |  |
|  |   - pypdf Text Extractor    |       |       - StandardScaler + OneHotEncoder|  |
|  |   - Keyword Matcher         | ----> |       - Random Forest Model           |  |
|  |   - SequenceMatcher Entity  |       |       - Calibrated Probability Output |  |
|  +-----------------------------+       +---------------------------------------+  |
|                                                            |                      |
|                                                            v                      |
|  +-----------------------------+       +---------------------------------------+  |
|  |   Explainability Engine     |       |       Deterministic Rule Engine       |  |
|  |   - Positive Signal Ranks   | <---- |       - 21 Statutory Criteria Checks  |  |
|  |   - Risk Signal Ranks       |       |       - Statutory Exclusion Override  |  |
|  |   - Natural Language Brief  |       |       - Review Priority Elevation     |  |
|  +-----------------------------+       +---------------------------------------+  |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                              GOVERNANCE & DATA TIER                               |
|                                                                                   |
|  +-----------------------------+       +---------------------------------------+  |
|  |   Review State Machine      |       |       Persistence Engine              |  |
|  |   - NOT REVIEWED            | ----> |       - SQLite Audit Database         |  |
|  |   - UNDER REVIEW            |       |         (`data/review_audit.db`)      |  |
|  |   - REVIEWED / ESCALATED    |       |       - Primary Mock Dataset (CSV)    |  |
|  +-----------------------------+       +---------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

---

## 7. Data Flow

```text
1. User submits / selects bid record and uploads statutory files (PDF/TXT)
2. Document Engine extracts text, verifies keywords, and calculates consistency
3. Document scores are transferred into the structured feature vector
4. Preprocessing pipeline normalizes numericals and one-hot encodes categoricals
5. Random Forest pipeline computes statistical compliance probability
6. Deterministic Rule Engine evaluates 21 statutory rules in parallel
7. If statutory exclusions exist (Debarred/Blacklisted), risk is escalated to High
8. Explainability engine extracts top signals and generates officer summary
9. Procurement Officer reviews findings, adds commentary, and transitions state
10. SQLite database records immutable audit log with officer attribution & timestamp
11. Dashboard consumes database records to update live procurement KPIs
```

---

## 8. Dataset Description

- **Source File:** [`data/bid_compliance_dataset.csv`](data/bid_compliance_dataset.csv)
- **Size:** `308,841` bytes | **Shape:** 1,000 rows × 34 columns
- **Class Balance:** 65.0% Compliant (`1`), 35.0% Non-Compliant / Review Required (`0`)
- **Nature:** Synthetic simulation generated to emulate real procurement conditions while protecting sensitive commercial data.

### Feature Breakdown
1. **Identifiers (Excluded from ML Training to prevent leakage):** `bid_id`, `tender_id`, `bidder_id`, `company_name`, `actual_compliance` (Target), `risk_level` (Excluded).
2. **Commercial & Categorical Features (18):** `project_type`, `state`, `district`, `gst_registration`, `gst_return_filing`, `pan_verification`, `udyam_registration`, `income_tax_compliance`, `startup_india_status`, `nsic_status`, `epfo_compliance`, `esic_compliance`, `oem_authorization`, `make_in_india_compliance`, `digilocker_verification`, `blacklist_status`, `debarment_status`, `tender_eligibility`.
3. **Numerical Features (10):** `bid_value_lakhs`, `local_content_percentage`, `required_documents`, `submitted_documents`, `document_completeness`, `documentation_consistency`, `previous_compliance_issues`, `portal_verification_issues`, `verification_delay_days`, `compliance_score`.

---

## 9. Exploratory Data Analysis (EDA) Summary

The dataset was analyzed via [`src/analyze_dataset.py`](src/analyze_dataset.py) and interactive notebooks:
- **Missingness:** 0 missing values across all 34 columns.
- **Uniqueness:** 1,000 distinct `bid_id` entries with zero row-level duplication.
- **Top Feature Correlates with Compliance:**
  - `compliance_score` ($r = +0.76$): Primary composite score indicator.
  - `verification_delay_days` ($r = -0.58$): Prolonged verification delays correlate strongly with non-compliance defaults.
  - `previous_compliance_issues` ($r = -0.52$): Past defaults indicate elevated risk.
  - `documentation_consistency` ($r = +0.48$): High cross-record consistency is a hallmark of genuine bidders.

---

## 10. Machine Learning Methodology

- **Framework:** Scikit-Learn
- **Feature Pipeline:**
  - `StandardScaler()` applied to all 10 numerical features.
  - `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` applied to 18 categoricals.
- **Data Partitioning:** 80/20 stratified split (`random_state=42`), allocating 800 training records and 200 testing records.
- **Hyperparameter Configuration:**
  - `n_estimators=100`, `max_depth=12`, `min_samples_split=5`, `min_samples_leaf=2`, `class_weight='balanced'`.

---

## 11. Model Comparison & Benchmarks

Three models were comparatively benchmarked on the test set:

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 90.50% | 0.9167 | 0.9402 | 0.9283 | 0.9423 |
| Decision Tree Classifier | 87.00% | 0.8983 | 0.8750 | 0.8865 | 0.8669 |
| **Random Forest Classifier (Selected)** | **91.00%** | **0.9062** | **0.9457** | **0.9211** | **0.9575** |

The Random Forest architecture was selected for deployment due to superior generalization, lower variance, and native feature importance extraction.

---

## 12. Prediction Engine & Statutory Rule Safety

The prediction engine in [`src/predict_compliance.py`](src/predict_compliance.py) computes compliance probability while enforcing deterministic safety overrides:
- **Statutory Exclusion Rules:**
  - `blacklist_status == 'Blacklisted'`
  - `debarment_status == 'Debarred'`
  - `tender_eligibility == 'Not Eligible'`
- **Override Logic:** If any statutory exclusion is active, the engine automatically:
  1. Sets `statutory_exclusions_flagged = True`.
  2. Overrides `risk_level` to `High`.
  3. Escalates `review_priority` to `HIGH PRIORITY REVIEW`.
  4. Triggers the warning banner in the user interface.
  5. Mandates human-in-the-loop review.

---

## 13. Document Verification Module

Implemented in [`src/document_verification.py`](src/document_verification.py):
- **Ingestion:** Supports PDF files (via `pypdf`) and plain-text files.
- **Keyword Dictionaries:** Maintained for 9 statutory document types (GSTIN, PAN, Udyam, OEM Authorization, EPFO, ESIC, etc.).
- **Consistency Analysis:** Utilizes `difflib.SequenceMatcher` to evaluate name similarity. Ratios below 60% generate major discrepancy warnings; ratios between 60% and 95% generate name variation notices.

---

## 14. Explainability Engine

Implemented in [`src/explainability.py`](src/explainability.py):
- Extracts top positive compliance factors (e.g., Class-I local content standing, verified tax records).
- Extracts top risk factors (e.g., documentation gaps, previous defaults).
- Compiles an executive summary with actionable guidance for procurement officers.
- Uses strictly non-causal attribution language.

---

## 15. Officer Review Workflow

Implemented in [`src/review_workflow.py`](src/review_workflow.py):
- **Lifecycle States:** `NOT REVIEWED`, `UNDER REVIEW`, `REVIEWED`, `ESCALATED`.
- **Supported Officer Actions:**
  - `REVIEW STARTED` $\rightarrow$ transitions to `UNDER REVIEW`.
  - `INFORMATION REQUESTED` $\rightarrow$ records clarification request; stays `UNDER REVIEW`.
  - `ADDITIONAL VERIFICATION REQUIRED` $\rightarrow$ flags need for field audit; stays `UNDER REVIEW`.
  - `REVIEW COMPLETED` $\rightarrow$ transitions to `REVIEWED`.
  - `ESCALATED FOR FURTHER REVIEW` $\rightarrow$ transitions to `ESCALATED`.
- **State Machine Protection:** Prevents invalid state jumps (e.g., `REVIEWED` bids cannot be reset to `UNDER REVIEW` without administrative authority).

---

## 16. Audit Trail Architecture

- **Database:** Local SQLite database at [`data/review_audit.db`](data/review_audit.db).
- **Tables:**
  - `bid_reviews`: Master record holding current status, latest action, and summary metrics.
  - `review_audit_log`: Append-only event store recording every status transition, action, remark, and timestamp.
- **Indexes:** Indexed on `bid_id` and `timestamp` to ensure instant historical log retrieval.

---

## 17. User Interface Modules

The 7-page interactive application is structured in [`app/app.py`](app/app.py):
1. **Dashboard:** High-level KPI metrics, risk distributions, state breakdowns, and review throughput.
2. **Bid Verification:** Real-time bidder evaluation, demo profile presets, probability visualization, and statutory rule audit.
3. **Document Verification:** Drag-and-drop file upload, automated text preview, keyword detection badges, and bid transfer.
4. **Risk Analysis:** Multi-dimensional filtering (State, Project Type, Risk Tier, Eligibility) with Plotly scatter and bar plots.
5. **Model Insights:** Feature importance rankings, algorithm comparison tables, and calibration insights.
6. **Officer Review:** Scrutiny console with workflow action triggers, remark submission, and real-time audit history.
7. **About System:** System architecture overview, CPCL hackathon context, and ethical AI governance standards.

---

## 18. Testing & Validation

The platform has been comprehensively verified:
- **Pytest Suite:** 43 automated unit and integration tests passing with 100% success rate across document parsing, explainability, prediction safety, and workflow transitions.
- **File Integrity:**
  - `models/compliance_model.pkl`: Exact size `1,259,122` bytes preserved.
  - `data/bid_compliance_dataset.csv`: Exact size `308,841` bytes preserved.
- **Terminology Verification:** Zero instances of unauthorized claims, automated decision keywords, or prohibited terminology.

---

## 19. Limitations

1. **Synthetic Procurement Data:** Evaluated exclusively on synthetic mock records; does not reflect real-world portal or CPCL transactions.
2. **Heuristic Document Analysis:** Keyword matching and sequence comparison are heuristic prototypes and do not replace certified forensic document examination.
3. **No External Government API Integration:** Operates offline and standalone without active connections to GSTN, MCA21, or public portal internal databases.
4. **Advisory Scope:** AI outputs represent decision support only; final procurement qualification decisions remain the sole responsibility of human officers.

---

## 20. Future Enhancements

1. **Optical Character Recognition (OCR):** Ingest scanned physical certificates and image-based PDFs.
2. **Authorized API Gateways:** Connect to official GSTN and procurement portal interfaces for automated validation.
3. **Transformer-Based Semantic Extraction:** Apply domain-specific NLP models to interpret complex technical specifications and tender clauses.
4. **Role-Based Access Control (RBAC):** Provide granular authentication across scrutiny officers, approving authorities, and vigilance committees.

---

## 21. Conclusion

The **AI-Powered Integrated Bid Compliance Verification Platform** successfully proves the feasibility of smart automation in public procurement. By harmonizing statistical machine learning with deterministic statutory rules and human-in-the-loop governance, the platform delivers high scrutiny efficiency while maintaining transparent, accountable, and legally sound procurement practices.
