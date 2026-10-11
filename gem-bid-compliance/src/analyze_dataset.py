"""
AI-Powered Integrated Bid Compliance Verification Platform for GeM Procurement
AIML Hackathon Problem Statement ID: 26100

Module: Exploratory Data Analysis (EDA)
File: src/analyze_dataset.py

Performs comprehensive Exploratory Data Analysis on the GeM bid compliance dataset:
- Basic Dataset Info & Metadata
- Data Quality & Integrity Checks
- Descriptive Statistics
- Target Analysis (actual_compliance)
- Risk Level Analysis (risk_level)
- Categorical Feature Distributions & Compliance Rates
- Numerical Feature Distributions & Outlier Detection
- Feature Relationships vs Target Variable
- Correlation Matrix & Heatmap
- Generates EDA summary report (data/processed/eda_summary.txt)
- Saves visualization artifacts to data/processed/
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


def load_dataset(file_path: str = "data/bid_compliance_dataset.csv") -> pd.DataFrame:
    """Loads the compliance dataset from CSV and verifies basic existence."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    df = pd.read_csv(path)
    return df


def perform_eda(df: pd.DataFrame, output_dir: str = "data/processed/") -> dict:
    """Runs the complete EDA workflow and produces charts and summary report."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    summary_lines = []

    def log(msg: str = ""):
        print(msg)
        summary_lines.append(msg)

    log("=" * 70)
    log("EXPLORATORY DATA ANALYSIS: GeM BID COMPLIANCE PLATFORM")
    log("AIML Hackathon Problem Statement ID: 26100")
    log("=" * 70)

    # ---------------------------------------------------------
    # 1. BASIC DATASET INFORMATION
    # ---------------------------------------------------------
    log("\n[1] BASIC DATASET INFORMATION")
    log("-" * 40)
    rows, cols = df.shape
    mem_usage_kb = df.memory_usage(deep=True).sum() / 1024
    log(f"Dataset Shape       : {df.shape}")
    log(f"Total Records (Rows): {rows}")
    log(f"Total Columns       : {cols}")
    log(f"Memory Usage        : {mem_usage_kb:.2f} KB")

    log("\nColumn Names and Data Types:")
    for col, dtype in df.dtypes.items():
        log(f" - {col:<30}: {dtype}")

    print("\n--- FIRST 5 ROWS ---")
    print(df.head(5).to_string())
    print("\n--- LAST 5 ROWS ---")
    print(df.tail(5).to_string())

    # ---------------------------------------------------------
    # 2. DATA QUALITY & INTEGRITY CHECKS
    # ---------------------------------------------------------
    log("\n[2] DATA QUALITY & INTEGRITY CHECKS")
    log("-" * 40)
    missing_series = df.isnull().sum()
    total_missing = missing_series.sum()
    log(f"Total Missing Values: {total_missing}")
    if total_missing > 0:
        log("Missing values per column:")
        for col, count in missing_series[missing_series > 0].items():
            log(f"  {col}: {count}")
    else:
        log("Zero missing values across all columns. Data integrity is intact.")

    dup_rows = df.duplicated().sum()
    dup_bids = df["bid_id"].duplicated().sum()
    unique_bidders = df["bidder_id"].nunique()
    unique_tenders = df["tender_id"].nunique()
    log(f"Duplicate Full Rows : {dup_rows}")
    log(f"Duplicate Bid IDs   : {dup_bids}")
    log(f"Unique Bidders      : {unique_bidders} (distinct vendors participating)")
    log(f"Unique Tenders      : {unique_tenders} (distinct tenders issued)")

    # Integrity Assertions / Validations
    val_score = ((df["compliance_score"] >= 0) & (df["compliance_score"] <= 100)).all()
    val_local = ((df["local_content_percentage"] >= 0) & (df["local_content_percentage"] <= 100)).all()
    val_comp = ((df["document_completeness"] >= 0) & (df["document_completeness"] <= 100)).all()
    val_cons = ((df["documentation_consistency"] >= 0) & (df["documentation_consistency"] <= 100)).all()
    val_docs = (df["submitted_documents"] <= df["required_documents"]).all()
    val_target = set(df["actual_compliance"].unique()).issubset({0, 1})
    val_risk = set(df["risk_level"].unique()).issubset({"Low", "Medium", "High"})

    log("\nDomain Integrity Assertions:")
    log(f" - compliance_score in [0, 100]        : {'PASSED' if val_score else 'FAILED'}")
    log(f" - local_content_percentage in [0, 100]: {'PASSED' if val_local else 'FAILED'}")
    log(f" - document_completeness in [0, 100]   : {'PASSED' if val_comp else 'FAILED'}")
    log(f" - documentation_consistency in [0, 100]: {'PASSED' if val_cons else 'FAILED'}")
    log(f" - submitted_docs <= required_docs     : {'PASSED' if val_docs else 'FAILED'}")
    log(f" - actual_compliance in {{0, 1}}        : {'PASSED' if val_target else 'FAILED'}")
    log(f" - risk_level in {{Low, Medium, High}}   : {'PASSED' if val_risk else 'FAILED'}")

    # ---------------------------------------------------------
    # 3. DESCRIPTIVE STATISTICS FOR NUMERICAL FEATURES
    # ---------------------------------------------------------
    log("\n[3] DESCRIPTIVE STATISTICS (NUMERICAL FEATURES)")
    log("-" * 40)
    num_cols = [
        "bid_value_lakhs",
        "local_content_percentage",
        "required_documents",
        "submitted_documents",
        "document_completeness",
        "documentation_consistency",
        "previous_compliance_issues",
        "portal_verification_issues",
        "verification_delay_days",
        "compliance_score"
    ]
    desc_df = df[num_cols].describe().T
    log(desc_df[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]].to_string())

    # ---------------------------------------------------------
    # 4. TARGET ANALYSIS (actual_compliance)
    # ---------------------------------------------------------
    log("\n[4] TARGET VARIABLE ANALYSIS (actual_compliance)")
    log("-" * 40)
    target_counts = df["actual_compliance"].value_counts().sort_index()
    target_pcts = df["actual_compliance"].value_counts(normalize=True).sort_index() * 100

    count_0 = target_counts.get(0, 0)
    count_1 = target_counts.get(1, 0)
    pct_0 = target_pcts.get(0, 0.0)
    pct_1 = target_pcts.get(1, 0.0)

    log(f"Class 0 (Non-Compliant / Requires Review): {count_0} records ({pct_0:.2f}%)")
    log(f"Class 1 (Compliant)                     : {count_1} records ({pct_1:.2f}%)")

    # Generate Target Distribution Bar Chart
    plt.figure(figsize=(7, 5))
    bars = plt.bar(
        ["0: Non-Compliant / Review", "1: Compliant"],
        [count_0, count_1],
        color=["#e63946", "#2a9d8f"],
        edgecolor="#264653",
        width=0.55
    )
    plt.title("Target Distribution (actual_compliance)", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Number of Bids", fontsize=11)
    plt.xlabel("Compliance Classification", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 10,
            f"{int(h)} ({h/rows*100:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold"
        )
    plt.ylim(0, max(count_0, count_1) + 80)
    plt.tight_layout()
    target_chart_path = out_path / "target_distribution.png"
    plt.savefig(target_chart_path, dpi=200)
    plt.close()
    log(f"Saved: {target_chart_path}")

    # ---------------------------------------------------------
    # 5. RISK LEVEL ANALYSIS (risk_level)
    # ---------------------------------------------------------
    log("\n[5] RISK LEVEL ANALYSIS")
    log("-" * 40)
    risk_order = ["Low", "Medium", "High"]
    risk_counts = df["risk_level"].value_counts().reindex(risk_order)
    risk_pcts = (risk_counts / rows) * 100

    for r_lvl in risk_order:
        log(f"{r_lvl:<10}: {risk_counts[r_lvl]} records ({risk_pcts[r_lvl]:.2f}%)")

    plt.figure(figsize=(7, 5))
    risk_colors = ["#2a9d8f", "#e9c46a", "#e76f51"]
    bars = plt.bar(risk_order, risk_counts, color=risk_colors, edgecolor="#264653", width=0.55)
    plt.title("Bidder Risk Level Distribution", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Number of Bids", fontsize=11)
    plt.xlabel("Risk Category", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 8,
            f"{int(h)} ({h/rows*100:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold"
        )
    plt.ylim(0, max(risk_counts) + 60)
    plt.tight_layout()
    risk_chart_path = out_path / "risk_distribution.png"
    plt.savefig(risk_chart_path, dpi=200)
    plt.close()
    log(f"Saved: {risk_chart_path}")

    # ---------------------------------------------------------
    # 6. CATEGORICAL FEATURE ANALYSIS & COMPLIANCE RATES
    # ---------------------------------------------------------
    log("\n[6] CATEGORICAL FEATURE VALUE COUNTS & COMPLIANCE RATES")
    log("-" * 40)
    cat_columns = [
        "project_type", "state", "gst_registration", "gst_return_filing",
        "pan_verification", "udyam_registration", "income_tax_compliance",
        "startup_india_status", "nsic_status", "epfo_compliance", "esic_compliance",
        "oem_authorization", "make_in_india_compliance", "digilocker_verification",
        "blacklist_status", "debarment_status", "tender_eligibility"
    ]

    for col in cat_columns:
        log(f"\n--- {col} ---")
        val_c = df[col].value_counts()
        comp_rate = df.groupby(col)["actual_compliance"].mean() * 100
        combo = pd.DataFrame({"Count": val_c, "Compliance Rate (%)": comp_rate.round(2)})
        log(combo.to_string())

    # Create multi-panel summary of key categorical features
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    key_cats = [
        "gst_registration", "gst_return_filing", "pan_verification",
        "tender_eligibility", "blacklist_status", "project_type"
    ]

    for idx, col in enumerate(key_cats):
        ax = axes[idx // 3, idx % 3]
        counts = df[col].value_counts()
        if col == "project_type":
            counts.plot(kind="barh", ax=ax, color="#457b9d", edgecolor="#1d3557")
            ax.set_title(f"Distribution of {col}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Count")
        else:
            counts.plot(kind="bar", ax=ax, color="#457b9d", edgecolor="#1d3557")
            ax.set_title(f"Distribution of {col}", fontsize=11, fontweight="bold")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y" if col != "project_type" else "x", linestyle="--", alpha=0.5)

    plt.suptitle("Key Categorical Features Distribution", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    cat_chart_path = out_path / "categorical_features.png"
    plt.savefig(cat_chart_path, dpi=200)
    plt.close()
    log(f"\nSaved: {cat_chart_path}")

    # ---------------------------------------------------------
    # 7. NUMERICAL FEATURE DISTRIBUTIONS
    # ---------------------------------------------------------
    log("\n[7] NUMERICAL FEATURE DISTRIBUTIONS")
    log("-" * 40)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    key_nums = [
        "compliance_score",
        "document_completeness",
        "documentation_consistency",
        "local_content_percentage",
        "verification_delay_days",
        "previous_compliance_issues",
        "portal_verification_issues",
        "bid_value_lakhs"
    ]

    for idx, col in enumerate(key_nums):
        ax = axes[idx // 4, idx % 4]
        ax.hist(df[col], bins=20, color="#3a86ff", edgecolor="#1e3d59", alpha=0.85)
        ax.set_title(f"{col}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Frequency")
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Distributions of Key Numerical Features", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    num_chart_path = out_path / "numerical_distributions.png"
    plt.savefig(num_chart_path, dpi=200)
    plt.close()
    log(f"Saved: {num_chart_path}")

    # ---------------------------------------------------------
    # 8. TARGET RELATIONSHIPS (Features vs actual_compliance)
    # ---------------------------------------------------------
    log("\n[8] FEATURE RELATIONSHIPS VS actual_compliance")
    log("-" * 40)
    features_to_compare = [
        "compliance_score",
        "document_completeness",
        "documentation_consistency",
        "previous_compliance_issues",
        "portal_verification_issues",
        "verification_delay_days"
    ]

    for feat in features_to_compare:
        grp = df.groupby("actual_compliance")[feat].agg(["mean", "median", "std"])
        log(f"\n{feat} by actual_compliance:")
        log(grp.round(2).to_string())

    # Generate Boxplot Grid for Features vs Target
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for idx, feat in enumerate(features_to_compare):
        ax = axes[idx // 3, idx % 3]
        data_0 = df[df["actual_compliance"] == 0][feat]
        data_1 = df[df["actual_compliance"] == 1][feat]

        bp = ax.boxplot(
            [data_0, data_1],
            tick_labels=["0: Non-Compliant", "1: Compliant"],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5)
        )
        bp["boxes"][0].set_facecolor("#ffb4a2")
        bp["boxes"][1].set_facecolor("#b7e4c7")

        ax.set_title(f"{feat} vs Target", fontsize=11, fontweight="bold")
        ax.set_ylabel(feat)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Feature Comparison Across Compliance Classes", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    target_rel_chart_path = out_path / "features_vs_target.png"
    plt.savefig(target_rel_chart_path, dpi=200)
    plt.close()
    log(f"Saved: {target_rel_chart_path}")

    # ---------------------------------------------------------
    # 9. CORRELATION ANALYSIS & HEATMAP
    # ---------------------------------------------------------
    log("\n[9] CORRELATION ANALYSIS (NUMERICAL FEATURES)")
    log("-" * 40)
    corr_cols = [
        "bid_value_lakhs",
        "required_documents",
        "submitted_documents",
        "document_completeness",
        "documentation_consistency",
        "local_content_percentage",
        "previous_compliance_issues",
        "portal_verification_issues",
        "verification_delay_days",
        "compliance_score",
        "actual_compliance"
    ]
    corr_matrix = df[corr_cols].corr()

    log("Correlation with actual_compliance (ranked by magnitude):")
    target_corr = corr_matrix["actual_compliance"].drop("actual_compliance").sort_values(ascending=False)
    for feat, val in target_corr.items():
        log(f" - {feat:<30}: {val:+.4f}")

    # Heatmap visualization
    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right", fontsize=9)
    plt.yticks(range(len(corr_cols)), corr_cols, fontsize=9)
    plt.title("Correlation Heatmap of Numerical Features", fontsize=13, fontweight="bold", pad=14)

    # Annotate numeric values inside cells
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            val = corr_matrix.iloc[i, j]
            text_color = "white" if abs(val) > 0.55 else "black"
            plt.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8)

    plt.tight_layout()
    heatmap_path = out_path / "correlation_heatmap.png"
    plt.savefig(heatmap_path, dpi=200)
    plt.close()
    log(f"Saved: {heatmap_path}")

    # ---------------------------------------------------------
    # 10. OUTLIER CHECK (IQR METHOD)
    # ---------------------------------------------------------
    log("\n[10] OUTLIER DETECTION (IQR METHOD)")
    log("-" * 40)
    log("Identified potential outliers without modifying synthetic baseline:")
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        log(
            f" - {col:<28}: {len(outliers):>3} outliers "
            f"(Bounds: [{lower_bound:.2f}, {upper_bound:.2f}])"
        )

    # ---------------------------------------------------------
    # 11. PLOTLY INTERACTIVE VISUALIZATION EXPORT
    # ---------------------------------------------------------
    log("\n[11] PLOTLY INTERACTIVE REPORT GENERATION")
    log("-" * 40)
    try:
        # Create an interactive multi-dimensional scatter plot with Plotly
        fig_plotly = px.scatter(
            df,
            x="compliance_score",
            y="documentation_consistency",
            color="actual_compliance",
            symbol="risk_level",
            size="bid_value_lakhs",
            hover_data=["bid_id", "company_name", "project_type", "gst_registration", "tender_eligibility"],
            title="Interactive Inspection: Compliance Score vs Consistency by Risk & Target",
            labels={
                "compliance_score": "Compliance Score (0-100)",
                "documentation_consistency": "Documentation Consistency (%)",
                "actual_compliance": "Compliance Target (0 or 1)",
                "risk_level": "Risk Category"
            },
            color_discrete_map={0: "#e63946", 1: "#2a9d8f"}
        )
        plotly_html_path = out_path / "interactive_eda_dashboard.html"
        fig_plotly.write_html(str(plotly_html_path))
        log(f"Saved interactive Plotly visual: {plotly_html_path}")
    except Exception as e:
        log(f"Note on Plotly export: {e}")

    # ---------------------------------------------------------
    # 12. WRITE SUMMARY TO eda_summary.txt
    # ---------------------------------------------------------
    summary_file_path = out_path / "eda_summary.txt"
    with open(summary_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    log(f"\nComplete EDA Summary saved to: {summary_file_path}")

    log("\n" + "=" * 70)
    log("EXPLORATORY DATA ANALYSIS COMPLETED SUCCESSFULLY")
    log("=" * 70)

    return {
        "shape": df.shape,
        "total_missing": total_missing,
        "duplicate_rows": dup_rows,
        "target_distribution": {0: count_0, 1: count_1},
        "risk_distribution": dict(risk_counts),
        "target_correlation": target_corr.to_dict(),
        "summary_file": str(summary_file_path)
    }


if __name__ == "__main__":
    dataset_file = "data/bid_compliance_dataset.csv"
    data = load_dataset(dataset_file)
    results = perform_eda(data)
