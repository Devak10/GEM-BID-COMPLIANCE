# AI-Powered Integrated Bid Compliance Verification Platform for GeM Procurement

[![AIML Hackathon](https://img.shields.io/badge/AIML%20Hackathon-Problem%20Statement%2026100-blue.svg)](https://gem.gov.in)
[![Organization](https://img.shields.io/badge/Organization-CPCL-orange.svg)](https://www.cpcl.co.in)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-51%20Passed-brightgreen.svg)]()
[![Governance](https://img.shields.io/badge/Governance-Human--in--the--Loop-purple.svg)]()

---

## 1. Project Overview

The **AI-Powered Integrated Bid Compliance Verification Platform** is an assistive decision-support prototype developed for the **AIML Hackathon** based on **Assigned Problem Statement ID: 26100**.

### The Challenge
Public procurement processes on platforms like Government e-Marketplace (GeM) require procurement officers to evaluate extensive bidder submissions against complex tender requirements, statutory criteria, financial qualifications, and supporting documentation. Evaluating dozens of bidders per tender is manual, time-intensive, and prone to oversight—especially across multi-source statutory records (GST, PAN, MSME/Udyam, labor compliance, and OEM authorizations).

### The Solution
This platform assists Procurement Officers by:
- **Organizing Bidder Information:** Aggregating disparate commercial, technical, and statutory data into a coherent evaluation profile.
- **Checking Prototype Compliance Rules:** Running 21 deterministic rule-based checks across tax standing, statutory filings, and tender criteria.
- **Verifying Uploaded Document Content:** Extracting text from PDF and TXT submissions to verify statutory keywords and detect cross-document naming discrepancies.
- **Predicting Compliance Probability:** Providing statistical compliance probability estimates using a trained Random Forest model.
- **Identifying Risk Indicators:** Classifying risk into transparent screening categories (`Low`, `Medium`, `High`).
- **Explaining Model Signals:** Breaking down prediction drivers into positive indicators and risk indicators using non-causal explainability.
- **Supporting Officer Review:** Providing a structured, state-machine-driven review interface (`NOT REVIEWED` $\rightarrow$ `UNDER REVIEW` $\rightarrow$ `REVIEWED` or `ESCALATED`).
- **Maintaining an Audit Trail:** Logging an immutable, chronological record of all officer actions, status transitions, and notes in an SQLite audit trail.

> **CRITICAL GOVERNANCE STATEMENT:**  
> **This is an AI-assisted decision-support prototype. Final procurement decisions remain strictly with authorized procurement officers. The platform does not make autonomous qualification or disqualification decisions.**

---

## 2. Problem Statement

- **Problem Statement ID:** 26100
- **Organization:** Chennai Petroleum Corporation Limited (CPCL)
- **Theme:** Smart Automation
- **Context:** AIML Hackathon

Procurement officers evaluating technical and commercial bids require smart automation tools to accelerate preliminary compliance screening, detect document deficiencies early, and ensure transparent, standardized, and auditable bid scrutiny.

---

## 3. Key Features

1. **Synthetic Procurement Dataset:** 1,000 comprehensive, realistic mock procurement records covering diverse sectors and statutory conditions.
2. **Exploratory Data Analysis (EDA):** In-depth distribution, correlation, and feature-target relationship analytics with visual artifacts.
3. **ML-Based Compliance Prediction:** Random Forest classifier predicting compliance probability with calibrated probabilities.
4. **Risk Assessment:** Multi-criteria risk scoring categorizing bids into `Low`, `Medium`, and `High` risk profiles.
5. **Prototype Rule-Based Compliance Checks:** 21 transparent, deterministic criteria verifying statutory registrations, domestic content, and portal defaults.
6. **AI-Assisted Document Verification:** Heuristic text extraction and parsing engine supporting PDF and TXT tender submissions.
7. **Cross-Document Consistency Checking:** Fuzzy sequence-matching algorithms checking name consistency and identifier concordance across document packs.
8. **AI Explainability:** Interpretable decision support highlighting positive compliance factors and risk indicators with non-causal phrasing.
9. **Procurement Officer Review Workflow:** Structured state machine enabling officers to start reviews, request clarifications, record remarks, and complete scrutiny.
10. **SQLite Audit Trail:** Immutable, timestamped logging of all review events, comments, and status transitions.
11. **Dashboard Analytics:** High-level procurement intelligence visualising compliance rates, risk distributions, state/sector metrics, and review throughput.
12. **Model Performance Analysis:** Comparative evaluation across multiple classification algorithms with ROC curves, confusion matrices, and feature importance tables.

---

## 4. System Architecture

```text
               Tender / Bid Submissions & Documents
                                ↓
                       Structured Bid Data
                                ↓
                  Document Verification Module
                   (Text Extraction & Checks)
                                ↓
                     Feature Preprocessing
                (StandardScaler + OneHotEncoder)
                                ↓
                     Random Forest ML Model
                                ↓
                     Compliance Probability
                                ↓
                 Rule-Based Compliance Checks (21 Rules)
                                ↓
                   Risk Assessment (Low/Med/High)
                                ↓
                      AI Explainability
                (Positive Signals & Risk Factors)
                                ↓
                  Procurement Officer Review
            (State Machine: Review, Remark, Escalate)
                                ↓
                   Immutable SQLite Audit Trail
```

### Architectural Separation of Duties
The platform enforces a strict separation between:
1. **The Machine Learning Layer:** Provides statistical pattern recognition and probability estimates based on learned feature relationships.
2. **The Deterministic Rule Layer:** Evaluates mandatory statutory constraints (such as debarment, blacklisting, and eligibility) independent of model confidence.
3. **The Human Governance Layer:** Grants sole decision-making authority to the Procurement Officer, backed by an immutable audit trail.

---

## 5. AI/ML Pipeline

The machine learning pipeline was trained and evaluated using scikit-learn:

1. **Data Preprocessing:** Handled via a unified `ColumnTransformer`:
   - *Numerical Features (10):* Scaled with `StandardScaler()`.
   - *Categorical Features (18):* Encoded with `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
2. **Dataset Partitioning:** 80/20 stratified train-test split (`random_state=42`) preserving compliance class distributions.
3. **Model Evaluation & Comparison:** Three classification algorithms were benchmarked on the synthetic test set:

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 90.50% | 0.9167 | 0.9402 | 0.9283 | 0.9423 |
| Decision Tree Classifier | 87.00% | 0.8983 | 0.8750 | 0.8865 | 0.8669 |
| **Random Forest Classifier (Selected)** | **91.00%** | **0.9062** | **0.9457** | **0.9211** | **0.9575** |

> **IMPORTANT EVALUATION NOTE:**  
> **These metrics reflect performance evaluated strictly on the synthetic hackathon demonstration dataset. They do not represent real-world GeM performance or production procurement accuracy.**

---

## 6. Dataset

- **Path:** [`data/bid_compliance_dataset.csv`](data/bid_compliance_dataset.csv)
- **File Size:** `308,841` bytes
- **Dimensions:** 1,000 rows × 34 columns
- **Target Variable:** `actual_compliance` (1 = Compliant, 0 = Non-Compliant / Requires Scrutiny)

### Representative Features
- **Bid Identifiers & Metadata:** `bid_id`, `tender_id`, `company_name`, `project_type`, `state`, `district`, `bid_value_lakhs`.
- **Statutory Tax & Registration Records:** `gst_registration`, `gst_return_filing`, `pan_verification`, `income_tax_compliance`, `digilocker_verification`.
- **Preferential & Technical Credentials:** `udyam_registration`, `startup_india_status`, `nsic_status`, `oem_authorization`, `make_in_india_compliance`, `local_content_percentage`.
- **Statutory Labor Compliance:** `epfo_compliance`, `esic_compliance`.
- **Central Integrity & Debarment Records:** `blacklist_status`, `debarment_status`, `tender_eligibility`.
- **Operational Submission Metrics:** `required_documents`, `submitted_documents`, `document_completeness`, `documentation_consistency`, `previous_compliance_issues`, `portal_verification_issues`, `verification_delay_days`, `compliance_score`.

> **Dataset Disclaimer:**  
> The dataset is purely synthetic mock data generated for this academic hackathon project. It contains zero actual CPCL or GeM commercial or vendor records.

---

## 7. Document Verification

- **Supported File Types:** PDF (`.pdf`) and Plain Text (`.txt`).
- **Core Functionalities:**
  - **Text Extraction:** Ingests tender PDFs via `pypdf` and text files with multi-encoding fallback.
  - **Automated Type Identification:** Matches extracted text against prototype keyword sets for 9 statutory document types.
  - **Keyword Verification:** Checks for presence of mandatory statutory phrases (e.g., GSTIN, Legal Name, MSME registration number).
  - **Completeness Computation:** Evaluates submitted vs. mandatory tender documents.
  - **Entity Consistency Checking:** Employs SequenceMatcher heuristic string comparison to detect discrepancies in vendor names or tax IDs across files.
  - **Bid Integration Transfer:** Transfers computed document completeness and consistency scores directly into the active Bid Verification workspace.

> **Document Verification Limitation:**  
> The document verification module uses prototype text extraction and heuristic checks. It does not constitute official statutory document validation or forensic fraud detection.

---

## 8. Risk Assessment

Bids are categorized into three prototype risk screening levels:
- 🟢 **LOW RISK:** Bid demonstrates high compliance probability ($\ge 75\%$), complete documentation ($\ge 90\%$), valid statutory registrations, and zero integrity flags.
- 🟡 **MEDIUM RISK:** Minor deficiencies detected (e.g., incomplete documentation 75–89%, slight consistency variations, or pending portal verifications).
- 🔴 **HIGH RISK:** Triggered by critical statutory exclusion flags (`Debarred`, `Blacklisted`, `Not Eligible`), low compliance probability ($< 50\%$), or major document discrepancies.

> **Clarification:**  
> These risk levels are prototype AI screening categories and are not official GeM or CPCL risk classifications. Critical statutory exclusion rules automatically elevate review priority to `HIGH PRIORITY REVIEW` regardless of model probability.

---

## 9. AI Explainability

The platform provides interpretable decision support grounded in feature importance without asserting causal relationships:
- **Positive Indicators:** Highlights verifiable factors that contributed positively to the compliance prediction (e.g., verified GST status, prompt verification turnaround, high local content).
- **Risk Indicators:** Pinpoints potential risk factors (e.g., missing documents, previous portal defaults, debarment flags).
- **Officer Summary:** Generates structured natural language briefs summarizing findings.
- **Review Priority:** Maps risk and model signals into transparent review tiers (`STANDARD REVIEW`, `PRIORITY REVIEW`, `HIGH PRIORITY REVIEW`).

> **Non-Causal Phrasing:**  
> All explanations clearly state that features *contributed to the model prediction* rather than asserting that any feature *caused* an outcome.

---

## 10. Human-in-the-Loop Review Workflow

The platform implements a formal state machine for procurement scrutiny:

```text
                  [ NOT REVIEWED ]
                         │
                         ▼ (REVIEW STARTED / INFO REQUESTED)
                  [ UNDER REVIEW ]
                   │            │
  (REVIEW COMPLETED)│            │(ESCALATED FOR FURTHER REVIEW)
                   ▼            ▼
              [ REVIEWED ]   [ ESCALATED ]
```

### Definition of Workflow Statuses
- **NOT REVIEWED:** Initial state. Bid is awaiting procurement officer evaluation.
- **UNDER REVIEW:** Officer has initiated scrutiny, requested information, or flagged additional verification requirements.
- **REVIEWED:** The Procurement Officer has completed the prototype review and recorded review findings. **This does NOT imply autonomous system approval or rejection.**
- **ESCALATED:** The bid requires committee evaluation, departmental legal review, or vendor clarification.

---

## 11. SQLite Audit Trail

- **Storage:** Local SQLite database at [`data/review_audit.db`](data/review_audit.db) (gitignored).
- **Audit Logging Schema:**
  - `bid_id`: Unique bid reference (e.g., `BID0001`).
  - `action`: Specific officer action taken (e.g., `REVIEW STARTED`, `INFORMATION REQUESTED`, `REVIEW COMPLETED`).
  - `previous_status`: Originating lifecycle state.
  - `new_status`: Resulting lifecycle state.
  - `remarks`: Officer commentary and audit notes.
  - `timestamp`: Chronological date and time stamp.
- **Immutability:** Audit records are strictly append-only; historical logs cannot be edited or deleted.

---

## 12. Technology Stack

- **Programming Language:** Python 3.10+
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (RandomForestClassifier, ColumnTransformer, OneHotEncoder, StandardScaler)
- **Model Serialization:** Joblib
- **Web Application & UI:** Streamlit
- **Data Visualization:** Plotly Graph Objects & Plotly Express
- **Document Processing:** PyPDF, Difflib
- **Database & Persistence:** SQLite3
- **Unit & Integration Testing:** Pytest

---

## 13. Project Structure

```text
gem-bid-compliance/
├── .env.example                     # Environment template
├── .gitignore                       # Git ignore configuration (ignores .env, *.db)
├── README.md                        # Primary project documentation
├── requirements.txt                 # Pinned project dependencies
├── app/
│   └── app.py                       # 7-page interactive Streamlit web application
├── data/
│   ├── bid_compliance_dataset.csv   # Primary synthetic dataset (1000 rows, 34 cols)
│   ├── demo/                        # Demo document samples for testing
│   ├── processed/                   # Precomputed benchmark artifacts and metrics
│   └── raw/                         # Raw data staging
├── docs/
│   ├── DEMO_GUIDE.md                # 5-minute timed presentation walkthrough
│   ├── FINAL_TECHNICAL_DOCUMENT.md  # Comprehensive technical specification
│   └── PRESENTATION_CONTENT.md      # Slide-by-slide hackathon presentation content
├── models/
│   ├── compliance_model.pkl         # Trained Random Forest pipeline (1,259,122 bytes)
│   └── model_features.txt           # List of 28 model feature names
├── notebooks/
│   └── exploratory_data_analysis.ipynb # Interactive EDA and visual analytics
├── scratch/
│   ├── verify_step10.py             # Step 10 regression verification harness
│   └── verify_step11.py             # Step 11 comprehensive system verification harness
├── src/
│   ├── __init__.py                  # Package initializer
│   ├── analyze_dataset.py           # Statistical analysis and EDA generation script
│   ├── create_dataset.py            # Synthetic dataset generator
│   ├── document_verification.py     # Document text extraction and verification engine
│   ├── explainability.py            # AI explainability and officer summary service
│   ├── predict_compliance.py        # Model inference and deterministic rule checks
│   ├── review_workflow.py           # SQLite audit trail and review state machine
│   └── train_model.py               # Model training, comparison, and serialization
├── tests/
│   ├── test_document_verification.py# Unit tests for document parsing (10 tests)
│   ├── test_explainability.py       # Unit tests for AI explainability (8 tests)
│   ├── test_predict_compliance.py   # Unit tests for prediction and rule safety (13 tests)
│   └── test_review_workflow.py      # Unit tests for review state machine (12 tests)
└── uploads/                         # Temporary workspace for uploaded documents
```

---

## 14. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed on your system.
- Git (optional, for cloning).

### 1. Clone or Navigate to the Repository
```powershell
# Clone the repository
git clone https://github.com/Devak10/GEM-BID-COMPLIANCE.git

# Navigate to the project directory
cd GEM-BID-COMPLIANCE/gem-bid-compliance
```

### 2. Create and Activate Virtual Environment
```powershell
# Create virtual environment
python -m venv venv

# Activate on Windows PowerShell
venv\Scripts\Activate.ps1
# Or on Windows Command Prompt
venv\Scripts\activate.bat
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 15. Running the Application

Launch the Streamlit web application using the virtual environment:

```powershell
venv\Scripts\streamlit.exe run app/app.py --server.port 8501
```

Once running, access the local dashboard at:
👉 **[http://localhost:8501](http://localhost:8501)**

---

## 16. Running the Test Suite

Execute the full pytest suite to validate all 51 tests:

```powershell
venv\Scripts\python.exe -m pytest -v
```

### Verified Test Outcome
```text
============================= test session starts =============================
platform win32 -- Python 3.14.2, pytest-9.1.1, pluggy-1.6.0
collected 51 items

tests/test_document_verification.py ...........                          [ 19%]
tests/test_explainability.py ........                                   [ 35%]
tests/test_predict_compliance.py .............                          [ 60%]
tests/test_review_workflow.py ............                              [ 84%]
tests/test_streamlit_app.py ........                                    [100%]

============================= 51 passed in 38.45s =============================
```

### Integration Verification Harness
To execute the comprehensive integration verification harness:
```powershell
venv\Scripts\python.exe scratch/verify_step11.py
```

---

## 17. Recommended Demo Workflow

Follow this 5-minute structured sequence for evaluation:

1. **Dashboard Overview:** Open `Dashboard` to showcase procurement KPIs, overall compliance rates, and risk distributions.
2. **Select Demo Profile:** Navigate to `Bid Verification` and pick an archetypal profile (e.g., `Incomplete Documentation (Metro Heavy Eng)`).
3. **Execute AI Verification:** Click **Run Compliance Verification** to view the ML prediction, probability bar, and risk classification.
4. **Inspect Statutory Rule Findings:** Review the deterministic rule check breakdown showing passed criteria and missing document warnings.
5. **Demonstrate Document Verification:** Navigate to `Document Verification`, upload sample documents (or select demo files), and view text extraction, keyword badges, and consistency scoring.
6. **Transfer Findings to Bid:** Click **Use Document Results for Bid Verification** to show dynamic updating of completeness metrics in the bid interface.
7. **Explainability & Review Priority:** Inspect the **AI Explainability** panel showing top positive indicators, risk factors, and the officer summary.
8. **Statutory Exclusion Safety Banner:** Select `Debarred Vendor (Pacific Electro-Mech)` or `Blacklisted Vendor (Pacific BuildTech)` to demonstrate that critical exclusions flag High Risk and display the safety warning banner even when general metrics are favorable.
9. **Officer Review & Audit Logging:** Click **Open Officer Review**, initiate review (`UNDER REVIEW`), add officer commentary, complete review (`REVIEWED`), and display the real-time SQLite audit trail.
10. **Return to Dashboard:** Return to `Dashboard` to show that the review metrics and completed count have updated dynamically.

---

## 18. Limitations

- **Synthetic Mock Dataset:** Built upon a synthetic 1,000-sample dataset; does not reflect real-world CPCL or GeM vendor transaction records.
- **Prototype Rule Engine:** Statutory thresholds (e.g., local content percentages, verification turnaround days) are configured as prototype approximations.
- **Heuristic Text Extraction:** Keyword identification and fuzzy name matching are heuristic implementations and do not include optical character recognition (OCR) for scanned images.
- **No Direct Government Portal Integration:** Operates standalone without external portal connectors or live government database connections.
- **Academic Demonstration Scope:** Designed specifically for hackathon evaluation; not certified for live production deployment.
- **Human Authority:** The system provides decision support only; final procurement qualification decisions remain the sole statutory prerogative of authorized procurement officers.

---

## 19. Ethical & Governance Design

- **Human-in-the-Loop Architecture:** All AI outputs are advisory. The system is structurally incapable of executing autonomous vendor qualification or disqualification.
- **Separation of Concerns:** Statistical ML inferences are decoupled from deterministic statutory exclusion rules. A high ML probability never suppresses an active debarment or blacklist indicator.
- **Transparent Decision Directives:** Explainability cards provide plain-language reasons for review prioritization.
- **Accountability & Auditability:** Every status change and comment is recorded in an immutable audit log with officer attribution and timestamps.
- **Ethical Language Standards:** Explanations strictly employ non-causal language to avoid false deterministic certainty.

---

## 20. Future Enhancements

The following extensions are identified for future real-world operationalization:
- **Optical Character Recognition (OCR):** Integration of Tesseract or Cloud Vision OCR to process scanned certificates, signed affidavits, and image-based PDFs.
- **Authorized Procurement System Bridges:** Secure, authenticated connectors to procurement portal interfaces, MCA21, and GSTN verification gateways.
- **Deep Domain NLP:** Transformer-based document parsing models (e.g., LayoutLM) for semantic extraction of complex financial statements and legal clauses.
- **Automated Model Monitoring:** Drift detection pipelines monitoring procurement trend shifts and feature distribution variations.
- **Role-Based Access Control (RBAC):** Multi-tier permission management for Junior Officers, Senior Procurement Managers, and Vigilance Committee members.
- **Configurable Policy Engine:** Dynamic administration console enabling procurement directors to adjust tender compliance rules and thresholds per category.

---

## 📄 Hackathon Metadata

- **Assigned Problem Statement:** 26100
- **Organization:** Chennai Petroleum Corporation Limited (CPCL)
- **Theme:** Smart Automation
- **Event:** AIML Hackathon
- **Prototype Status:** Prototype Complete & Hackathon Demo Ready
