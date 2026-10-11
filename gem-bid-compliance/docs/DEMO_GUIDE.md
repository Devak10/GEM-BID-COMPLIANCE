# 5-MINUTE HACKATHON DEMONSTRATION GUIDE

**Project:** AI-Powered Integrated Bid Compliance Verification Platform for GeM Procurement  
**Problem Statement ID:** 26100 | **Theme:** Smart Automation | **Organization:** CPCL  
**Event:** AIML Hackathon  
**Target Duration:** Exactly 5 Minutes (0:00 – 5:00)  

---

## Quick Setup Before Presenting

1. Ensure the Streamlit application is running:
   ```powershell
   venv\Scripts\streamlit.exe run app/app.py --server.port 8501
   ```
2. Open your web browser to: **`http://localhost:8501`**
3. Maximize the browser window for optimal presentation.

---

## ⏱️ Minute-by-Minute Demonstration Script

---

### [0:00 – 0:30] Phase 1: Problem Introduction & Context

- **What to Click:** Keep the browser on the **Dashboard** page.
- **What to Show:** Point out the title header: *"AI-Powered Integrated Bid Compliance Verification Platform"*, referencing Problem Statement 26100 for CPCL.
- **What to Say:**
  > *"Respected Evaluators, in public procurement on platforms like GeM, Procurement Officers must evaluate dozens of bidder submissions per tender. Verifying tax documents, MSME statuses, OEM authorizations, and debarment lists manually is slow, repetitive, and vulnerable to human oversight.*  
  > *We have developed an AI-assisted decision-support prototype that combines Machine Learning, deterministic statutory rules, and document verification into a unified platform—keeping the authorized Procurement Officer in complete control."*
- **Key Terminology:** *Decision-Support Assistant*, *Human-in-the-Loop*, *Statutory Scrutiny*.

---

### [0:30 – 1:15] Phase 2: High-Level Procurement Dashboard

- **What to Click:** Sidebar $\rightarrow$ **Dashboard** (already visible).
- **What to Show:**
  - The top 4 KPI cards: Total Bids (1,000), Overall Compliance Rate (65%), High-Risk Submissions, and Active Review Status.
  - The **Risk Distribution** bar chart and the **Sector Breakdown** chart.
- **What to Say:**
  > *"Our Dashboard gives senior procurement authorities instant situational awareness across active tenders. In our synthetic demonstration dataset of 1,000 tenders, we can immediately identify high-risk bids, track compliance rates across sectors like Infrastructure and IT Services, and monitor the live progress of officer reviews in real time."*
- **Key Terminology:** *Screening Distribution*, *Procurement Telemetry*, *Tender Overview*.

---

### [1:15 – 2:15] Phase 3: Bid Verification & Statutory Rule Layer

- **What to Click:** Sidebar $\rightarrow$ **Bid Verification**.
- **What to Show:**
  - The **Quick Demo Profiles** dropdown.
  - Select: `Incomplete Documentation (Metro Heavy Eng)`.
  - Click the **Run Compliance Verification** button.
  - Point to the AI Prediction Card: *Predicted Compliance: Compliant (1)*, *Probability: 83.8%*, *Risk Level: Medium*.
  - Scroll down to the **Basic Statutory & Rule Checks (21 Rules)** expandable container.
- **What to Say:**
  > *"Here on the Bid Verification console, officers can evaluate incoming bids instantly. When we run verification on Metro Heavy Engineering, our Random Forest model identifies an 83.8% compliance probability, categorizing it as Medium Risk.*  
  > *Notice that right below the ML output, our deterministic Rule Engine evaluates 21 distinct statutory criteria—clearly highlighting that while GST and PAN are verified, document completeness is only 75% due to missing technical annexures."*
- **Key Terminology:** *Calibrated Compliance Probability*, *Deterministic Statutory Rules*, *Independent Safety Checks*.

---

### [2:15 – 3:00] Phase 4: Document Verification & Integration Transfer

- **What to Click:** Sidebar $\rightarrow$ **Document Verification**.
- **What to Show:**
  - Point to the Upload section. (You can click **Load Demo Document Package**).
  - Show the **Extracted Text Preview** and the green **✓ Passed** keyword badge for GSTIN and Legal Name.
  - Highlight the **Document Consistency Score (95%)**.
  - Click the green button: **Use Document Results for Bid Verification**.
- **What to Say:**
  > *"Bidders submit physical or digital certificates. Our Document Verification module ingests PDFs and text files, extracts content, verifies statutory keywords, and checks entity consistency across files.*  
  > *With one click—'Use Document Results'—the verified document completeness metrics are transferred directly into the bid evaluation session without altering any underlying statutory tax data."*
- **Key Terminology:** *Keyword Verification*, *Cross-Document Consistency*, *Seamless Data Transfer*.

---

### [3:00 – 4:00] Phase 5: AI Explainability & Critical Statutory Exclusion Safety

- **What to Click:**
  1. Return to **Bid Verification**.
  2. Scroll to the **AI Explainability & Compliance Intelligence** section.
  3. Then switch Demo Profile to: `Debarred Vendor (Pacific Electro-Mech)`.
  4. Click **Run Compliance Verification**.
- **What to Show:**
  - Show the **Top Positive Factors** vs. **Key Risk Factors** cards.
  - Point to the **Procurement Officer Executive Brief**.
  - On the Debarred profile, show the **CRITICAL STATUTORY EXCLUSION SAFETY BANNER**:
    > *"ML model indicates high compliance probability, but critical rule-based exclusion indicators require officer verification."*
- **What to Say:**
  > *"Explainability is essential for trust. Our platform breaks down the model's signals into positive indicators and risk factors using non-causal language.*  
  > *Now notice our critical rule safety feature: Even if a vendor exhibits favorable commercial scores, if our rule layer detects an active Debarment or Blacklist entry, the system instantly escalates the risk to High, elevates priority to High Priority Review, and displays this prominent warning banner. The ML model is never allowed to override a statutory exclusion."*
- **Key Terminology:** *Non-Causal Explainability*, *Statutory Exclusion Override*, *Ethical AI Governance*.

---

### [4:00 – 4:40] Phase 6: Officer Review Workflow & SQLite Audit Trail

- **What to Click:**
  1. Scroll down and click **Open Officer Review** (or navigate via sidebar to **Officer Review**).
  2. In the Action dropdown, select: `REVIEW STARTED` $\rightarrow$ Click **Submit Review Action**.
  3. In remarks, type: *"Verified supplementary certificate; no debarment found."* $\rightarrow$ Select `REVIEW COMPLETED` $\rightarrow$ Click **Submit Review Action**.
  4. Scroll to the **Immutable Review Audit Trail** table.
- **What to Show:**
  - The status changing from `NOT REVIEWED` $\rightarrow$ `UNDER REVIEW` $\rightarrow$ `REVIEWED`.
  - The chronological audit log showing timestamps, officer action, and commentary.
- **What to Say:**
  > *"Every review follows a strict state machine. The officer initiates scrutiny, requests clarification, or completes the review.*  
  > *Every single transition and remark is permanently stored in an immutable SQLite audit trail. 'REVIEWED' means the human officer has completed scrutiny and recorded findings—it does not mean the system autonomously approved or rejected the bid."*
- **Key Terminology:** *State Machine Workflow*, *Immutable Audit Trail*, *Human Accountability*.

---

### [4:40 – 5:00] Phase 7: Conclusion & Summary

- **What to Click:** Sidebar $\rightarrow$ **Dashboard** (to show updated review counts) or **About System**.
- **What to Show:** Point out the updated review metric card on the Dashboard.
- **What to Say:**
  > *"In summary, our platform provides Chennai Petroleum Corporation Limited with a smart automation tool that reduces bid scrutiny time from hours to minutes, prevents missed compliance defaults, and provides full auditability—while strictly preserving human procurement governance.*  
  > *Thank you. We welcome your questions."*
- **Key Terminology:** *Smart Automation*, *CPCL Smart Governance*, *Audit-Ready Procurement*.

---

## 🎯 Important Tips for Evaluator Questions

| Question | Recommended Answer |
| :--- | :--- |
| **"Does the AI approve or reject the bid automatically?"** | *"No. The system strictly provides decision support. The final qualification or disqualification decision remains 100% with the authorized Procurement Officer."* |
| **"What if the ML model gives a 99% probability to a blacklisted bidder?"** | *"Our architecture decouples ML from statutory rules. Any active debarment or blacklist indicator automatically overrides the risk rating to High and mandates officer scrutiny, regardless of ML probability."* |
| **"What data was used to train the model?"** | *"We generated a balanced, synthetic dataset of 1,000 procurement bids modeled on typical GeM tender criteria, as real government bidding data is confidential."* |
| **"Can it handle scanned document images?"** | *"Currently, our prototype extracts text from digital PDFs and plain text files. As part of our documented future scope, we plan to integrate OCR engines like Tesseract for scanned paper submissions."* |
