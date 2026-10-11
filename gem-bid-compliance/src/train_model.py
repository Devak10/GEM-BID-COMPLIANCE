"""
AI-Powered Integrated Bid Compliance Verification Platform for GeM Procurement
AIML Hackathon Problem Statement ID: 26100
Organization: Chennai Petroleum Corporation Limited (CPCL)
Theme: Smart Automation

Module: Machine Learning Model Training & Evaluation
File: src/train_model.py

Builds and evaluates machine-learning classification models to predict bidder compliance:
- Target: actual_compliance (0 = Non-Compliant / Requires Review, 1 = Compliant)
- Preprocessing: ColumnTransformer with OneHotEncoder for categoricals and StandardScaler for numericals
- Models: Logistic Regression, Decision Tree Classifier, Random Forest Classifier
- Metrics: Accuracy, Precision, Recall, F1 Score, Confusion Matrix, Classification Report
- Saves: Preprocessing + Model Pipeline, Comparison Table, Confusion Matrices, Feature Importance, Summary Report
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def run_training_pipeline(
    data_path: str = "data/bid_compliance_dataset.csv",
    output_dir: str = "data/processed",
    models_dir: str = "models",
    random_state: int = 42
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    mod_path = Path(models_dir)
    mod_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 4: MACHINE LEARNING MODEL TRAINING & EVALUATION")
    print("AIML Hackathon Problem Statement ID: 26100 | GeM Procurement Compliance Platform")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------------------
    df = pd.read_csv(data_path)
    print(f"\n[1] DATASET LOADED: {data_path}")
    print(f" - Shape: {df.shape} ({df.shape[0]} rows, {df.shape[1]} columns)")
    print(f" - Features before preprocessing: {df.shape[1] - 1} (excluding target)")
    print(" - Target Distribution (actual_compliance):")
    t_counts = df["actual_compliance"].value_counts().sort_index()
    t_pcts = df["actual_compliance"].value_counts(normalize=True).sort_index() * 100
    for cls_val in [0, 1]:
        label_str = "0 (Non-Compliant / Requires Review)" if cls_val == 0 else "1 (Compliant)"
        print(f"   * Class {label_str}: {t_counts[cls_val]} ({t_pcts[cls_val]:.2f}%)")

    # -------------------------------------------------------------
    # 2. FEATURE SELECTION & PREPARATION
    # -------------------------------------------------------------
    # Excluded columns: identifiers and target leakage
    excluded_columns = [
        "bid_id",
        "tender_id",
        "bidder_id",
        "company_name",
        "risk_level",  # Derived classification; excluded to prevent leakage
        "actual_compliance"  # Target
    ]

    feature_cols = [c for c in df.columns if c not in excluded_columns]
    target_col = "actual_compliance"

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Identify categorical vs numerical columns
    cat_features = [
        "project_type", "state", "district", "gst_registration", "gst_return_filing",
        "pan_verification", "udyam_registration", "income_tax_compliance",
        "startup_india_status", "nsic_status", "epfo_compliance", "esic_compliance",
        "oem_authorization", "make_in_india_compliance", "digilocker_verification",
        "blacklist_status", "debarment_status", "tender_eligibility"
    ]

    num_features = [
        "bid_value_lakhs", "local_content_percentage", "required_documents",
        "submitted_documents", "document_completeness", "documentation_consistency",
        "previous_compliance_issues", "portal_verification_issues",
        "verification_delay_days", "compliance_score"
    ]

    print(f"\n[2] FEATURE PREPARATION")
    print(f" - Excluded columns ({len(excluded_columns)}): {excluded_columns}")
    print(f" - Total ML input features: {len(feature_cols)}")
    print(f"   * Categorical features ({len(cat_features)}): {cat_features}")
    print(f"   * Numerical features ({len(num_features)}): {num_features}")

    # Build ColumnTransformer Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
        ],
        remainder="drop"
    )

    # -------------------------------------------------------------
    # 3. TRAIN / TEST SPLIT (80% Train, 20% Test, Stratified)
    # -------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=random_state,
        stratify=y
    )

    print(f"\n[3] TRAIN / TEST SPLIT (80/20 Stratified)")
    print(f" - Training Set : {X_train.shape[0]} samples, {X_train.shape[1]} raw features")
    print(f" - Testing Set  : {X_test.shape[0]} samples, {X_test.shape[1]} raw features")
    print(f" - Train Class 0: {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%), Class 1: {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")
    print(f" - Test Class 0 : {(y_test == 0).sum()} ({(y_test == 0).mean()*100:.1f}%), Class 1: {(y_test == 1).sum()} ({(y_test == 1).mean()*100:.1f}%)")

    # -------------------------------------------------------------
    # 4. DEFINE CLASSIFICATION CANDIDATES
    # -------------------------------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state, max_depth=8, min_samples_leaf=3),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=random_state)
    }

    # -------------------------------------------------------------
    # 5. TRAIN, PREDICT & EVALUATE
    # -------------------------------------------------------------
    print(f"\n[4] MODEL TRAINING & EVALUATION")
    results = []
    trained_pipelines = {}
    reports_text = []

    for name, clf in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1)
        rec = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)

        # Class 0 specific metrics for compliance review capability
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, target_names=["0: Non-Compliant", "1: Compliant"])

        trained_pipelines[name] = pipeline
        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4),
            "cm": cm,
            "cr": cr
        })

        reports_text.append(f"==================================================")
        reports_text.append(f"CLASSIFICATION REPORT: {name}")
        reports_text.append(f"==================================================")
        reports_text.append(cr)
        reports_text.append(f"Confusion Matrix (Actual Rows x Predicted Cols):\n{cm}\n\n")

    # -------------------------------------------------------------
    # 6. MODEL COMPARISON
    # -------------------------------------------------------------
    comp_df = pd.DataFrame(results)[["Model", "Accuracy", "Precision", "Recall", "F1 Score"]]
    print("\n--- Model Evaluation Comparison ---")
    print(comp_df.to_string(index=False))

    comparison_csv_path = out_path / "model_comparison.csv"
    comp_df.to_csv(comparison_csv_path, index=False)
    print(f"\nSaved comparison table to: {comparison_csv_path}")

    # Save classification reports text
    reports_file_path = out_path / "classification_reports.txt"
    with open(reports_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(reports_text))
    print(f"Saved classification reports to: {reports_file_path}")

    # -------------------------------------------------------------
    # 7. CONFUSION MATRIX VISUALIZATIONS
    # -------------------------------------------------------------
    file_slugs = {
        "Logistic Regression": "confusion_matrix_logistic_regression.png",
        "Decision Tree": "confusion_matrix_decision_tree.png",
        "Random Forest": "confusion_matrix_random_forest.png"
    }

    for item in results:
        name = item["Model"]
        cm = item["cm"]
        slug = file_slugs[name]

        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.matshow(cm, cmap="Blues", alpha=0.85)
        fig.colorbar(cax, fraction=0.046, pad=0.04)

        labels = ["0: Non-Compliant", "1: Compliant"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)

        # Text inside cells
        for i in range(2):
            for j in range(2):
                cell_val = cm[i, j]
                ax.text(
                    j, i, f"{cell_val}",
                    ha="center", va="center",
                    color="white" if cell_val > (cm.max() / 2) else "black",
                    fontsize=14, fontweight="bold"
                )

        ax.set_xlabel("Predicted", fontsize=11, fontweight="bold", labelpad=10)
        ax.set_ylabel("Actual", fontsize=11, fontweight="bold", labelpad=10)
        ax.set_title(f"Confusion Matrix: {name}", fontsize=12, fontweight="bold", pad=16)
        plt.tight_layout()

        cm_path = out_path / slug
        plt.savefig(cm_path, dpi=200)
        plt.close()
        print(f"Saved: {cm_path}")

    # -------------------------------------------------------------
    # 8. MODEL SELECTION & RATIONALE
    # -------------------------------------------------------------
    # In public procurement compliance verification:
    # High sensitivity to detecting non-compliant bids (Class 0 recall) and high precision (avoiding false rejections)
    # Balanced F1 Score on the test set is key.
    # We inspect the F1 score and overall balance to choose the champion model.
    best_row = comp_df.sort_values(by="F1 Score", ascending=False).iloc[0]
    selected_model_name = best_row["Model"]
    selected_pipeline = trained_pipelines[selected_model_name]

    print("\n" + "=" * 70)
    print(f"PROTOTYPE MODEL SELECTION: {selected_model_name}")
    print("=" * 70)
    print(f"Selection Rationale based on empirical test metrics:")
    print(f" - Accuracy  : {best_row['Accuracy']*100:.2f}%")
    print(f" - Precision : {best_row['Precision']*100:.2f}%")
    print(f" - Recall    : {best_row['Recall']*100:.2f}%")
    print(f" - F1 Score  : {best_row['F1 Score']*100:.2f}%")
    print(
        f"The {selected_model_name} demonstrated the most balanced trade-off between identifying "
        f"genuinely compliant bids (high precision) while maintaining strong detection of non-compliant "
        f"or high-risk bids without erratic variance.\n"
        f"IMPORTANT NOTE: The model serves as an intelligent assistive scoring tool. "
        f"The final qualification or disqualification decision strictly rests with the GeM Procurement Officer."
    )

    # -------------------------------------------------------------
    # 9. SAVE THE SELECTED PIPELINE & FEATURE LIST
    # -------------------------------------------------------------
    model_save_path = mod_path / "compliance_model.pkl"
    joblib.dump(selected_pipeline, model_save_path)
    print(f"\nSaved full preprocessing + model pipeline to: {model_save_path}")

    # Save feature names documentation
    features_doc_path = mod_path / "model_features.txt"
    with open(features_doc_path, "w", encoding="utf-8") as f:
        f.write("AI-POWERED GeM BID COMPLIANCE PLATFORM - MODEL INPUT FEATURES\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Selected Champion Model: {selected_model_name}\n")
        f.write(f"Target Column: {target_col} (0 = Non-Compliant / Review, 1 = Compliant)\n\n")
        f.write(f"1. Raw Numerical Features ({len(num_features)}):\n")
        for f_name in num_features:
            f.write(f"   - {f_name}\n")
        f.write(f"\n2. Raw Categorical Features ({len(cat_features)}):\n")
        for f_name in cat_features:
            f.write(f"   - {f_name}\n")
        f.write(f"\n3. Excluded Columns ({len(excluded_columns)}):\n")
        for f_name in excluded_columns:
            f.write(f"   - {f_name}\n")
    print(f"Saved feature documentation to: {features_doc_path}")

    # -------------------------------------------------------------
    # 10. FEATURE IMPORTANCE ANALYSIS
    # -------------------------------------------------------------
    prepr = selected_pipeline.named_steps["preprocessor"]
    transformed_feature_names = prepr.get_feature_names_out()
    classifier_step = selected_pipeline.named_steps["classifier"]

    if hasattr(classifier_step, "feature_importances_"):
        importances = classifier_step.feature_importances_
        metric_label = "Gini Importance"
    elif hasattr(classifier_step, "coef_"):
        importances = np.abs(classifier_step.coef_[0])
        metric_label = "Absolute Coefficient Magnitude"
    else:
        importances = np.zeros(len(transformed_feature_names))
        metric_label = "Importance"

    fi_df = pd.DataFrame({
        "Feature": transformed_feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    fi_csv_path = out_path / "feature_importance.csv"
    fi_df.to_csv(fi_csv_path, index=False)
    print(f"\nSaved feature importance table to: {fi_csv_path}")

    print("\n--- Top 15 Most Important Features ---")
    print(fi_df.head(15).to_string(index=False))

    # Feature Importance Plot
    top15 = fi_df.head(15).iloc[::-1]  # Invert for horizontal bar plot
    plt.figure(figsize=(10, 7))
    plt.barh(top15["Feature"], top15["Importance"], color="#2a9d8f", edgecolor="#264653")
    plt.title(f"Top 15 Predictive Features ({selected_model_name} - {metric_label})", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel(metric_label, fontsize=10)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fi_img_path = out_path / "feature_importance.png"
    plt.savefig(fi_img_path, dpi=200)
    plt.close()
    print(f"Saved feature importance chart to: {fi_img_path}")

    # -------------------------------------------------------------
    # 11. MODEL SUMMARY REPORT (data/processed/ml_summary.txt)
    # -------------------------------------------------------------
    summary_text = f"""======================================================================
MACHINE LEARNING MODEL SUMMARY: GeM BID COMPLIANCE PLATFORM
AIML Hackathon Problem Statement ID: 26100 | Smart Automation
======================================================================

1. DATASET & PROBLEM OVERVIEW
----------------------------------------
- Total Dataset Size: {df.shape[0]} records x {df.shape[1]} columns
- Target Variable   : actual_compliance (Binary: 0 = Non-Compliant / Requires Review, 1 = Compliant)
- Target Distribution:
  * Class 0 (Non-Compliant / Review): {t_counts[0]} ({t_pcts[0]:.2f}%)
  * Class 1 (Compliant)            : {t_counts[1]} ({t_pcts[1]:.2f}%)

2. FEATURE SELECTION & PREPROCESSING
----------------------------------------
- Excluded Features ({len(excluded_columns)}):
  * Identifiers: bid_id, tender_id, bidder_id, company_name
  * Target Leakage Prevention: risk_level (derived risk label)
- Input Features ({len(feature_cols)}):
  * Numerical ({len(num_features)}): {num_features}
  * Categorical ({len(cat_features)}): {cat_features}
- Preprocessing Pipeline:
  * Numerical  : StandardScaler()
  * Categorical: OneHotEncoder(handle_unknown='ignore', sparse_output=False)
- Total Transformed Features after Encoding: {len(transformed_feature_names)}

3. TRAIN / TEST SPLIT
----------------------------------------
- Split Ratio : 80% Training ({X_train.shape[0]} samples), 20% Testing ({X_test.shape[0]} samples)
- Stratification: Enabled on actual_compliance
- Random State: {random_state}

4. CANDIDATE MODELS & EVALUATION METRICS
----------------------------------------
{comp_df.to_string(index=False)}

5. SELECTED PROTOTYPE MODEL & RATIONALE
----------------------------------------
- Selected Champion Model: {selected_model_name}
- Metrics on Test Set:
  * Accuracy : {best_row['Accuracy']*100:.2f}%
  * Precision: {best_row['Precision']*100:.2f}%
  * Recall   : {best_row['Recall']*100:.2f}%
  * F1 Score : {best_row['F1 Score']*100:.2f}%
- Rationale:
  The {selected_model_name} provides superior balance between high precision (minimizing false disqualification of compliant vendors) and high recall (detecting non-compliant submissions), outperforming the baseline models on composite F1 score.

6. TOP 10 INFLUENTIAL FEATURES
----------------------------------------
{fi_df.head(10).to_string(index=False)}
(Note: Feature importance indicates mathematical association in the synthetic data, not causal proof.)

7. SAVED ARTIFACTS
----------------------------------------
- Trained Model Pipeline: models/compliance_model.pkl
- Model Feature Spec    : models/model_features.txt
- Model Comparison CSV  : data/processed/model_comparison.csv
- Classification Reports: data/processed/classification_reports.txt
- Feature Importance CSV: data/processed/feature_importance.csv
- Feature Importance PNG: data/processed/feature_importance.png
- Confusion Matrices    :
  * data/processed/confusion_matrix_logistic_regression.png
  * data/processed/confusion_matrix_decision_tree.png
  * data/processed/confusion_matrix_random_forest.png

8. PROTOTYPE DISCLAIMER & LIMITATIONS
----------------------------------------
"This model is trained on synthetic/demo data and is intended only as a prototype for the Hackathon solution. It must not be treated as an official government or GeM compliance decision system."

Actual production deployment requires:
- Validated real-world procurement datasets.
- Official API integrations with authoritative portals (MCA-21, GSTN, Udyam, DigiLocker, CPPP).
- Rule-based hard statutory guardrails (e.g. mandatory disqualification for blacklisted vendors).
- Strict cybersecurity and privacy compliance.
- Final decision authority strictly reserved for the Procurement Officer.
"""
    ml_summary_path = out_path / "ml_summary.txt"
    with open(ml_summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"Saved ML summary report to: {ml_summary_path}")

    # -------------------------------------------------------------
    # 12. MODEL VALIDATION CHECKS
    # -------------------------------------------------------------
    print("\n[5] MODEL VALIDATION CHECKS")
    print("-" * 40)
    # Check 1: No missing values in inputs
    assert X.isna().sum().sum() == 0, "Validation Error: Missing values in features!"
    print(" [PASSED] No missing values entered the model.")

    # Check 2: Target contains only 0 and 1
    assert set(y.unique()).issubset({0, 1}), "Validation Error: Target not in {0, 1}!"
    print(" [PASSED] Target variable strictly contains only {0, 1}.")

    # Check 3: Stratified split verification
    train_prop = (y_train == 1).mean()
    test_prop = (y_test == 1).mean()
    assert abs(train_prop - test_prop) < 0.02, "Validation Error: Train/test split not stratified!"
    print(f" [PASSED] Train/test split stratified: Train pos={train_prop:.3f}, Test pos={test_prop:.3f}.")

    # Check 4: Identifiers and risk_level not in features
    for col in excluded_columns:
        assert col not in X.columns, f"Validation Error: Excluded column {col} found in features!"
    print(" [PASSED] Identifiers and risk_level successfully excluded.")

    # Check 5: Model file exists
    assert model_save_path.exists(), "Validation Error: Model pkl file missing!"
    print(" [PASSED] Serialized pipeline file exists on disk.")

    # Check 6: Load model and predict on test sample
    loaded_pipeline = joblib.load(model_save_path)
    sample_test = X_test.head(5)
    sample_preds = loaded_pipeline.predict(sample_test)
    sample_probs = loaded_pipeline.predict_proba(sample_test)
    assert len(sample_preds) == 5, "Validation Error: Loaded model failed to make predictions!"
    print(" [PASSED] Saved model loaded successfully with joblib and verified on test samples.")
    print("          Sample predictions:", sample_preds)
    print("          Sample probabilities:\n", np.round(sample_probs, 3))

    print("\n" + "=" * 70)
    print("STEP 4: MACHINE LEARNING MODEL PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return comp_df, selected_model_name


if __name__ == "__main__":
    run_training_pipeline()
