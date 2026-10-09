"""
Data Cleaning & Visualization Dashboard
=======================================
A modern, professional Streamlit web application for:
- Automated data hygiene (Missing values, Duplicates, IQR Outliers)
- Before & After dataset audits
- Interactive data visualizations (Plotly, Seaborn, Matplotlib)
- Dynamic, data-driven key findings and executive insights
"""

import os
from typing import List, Optional
import streamlit as st
import pandas as pd
import numpy as np

# Internal modules
from src.data_cleaning import (
    get_dataset_summary,
    get_missing_value_table,
    get_duplicate_rows,
    detect_outliers_iqr,
    clean_dataset,
)
from src.visualizations import (
    plot_numeric_histogram,
    plot_categorical_bar,
    plot_box_plots,
    plot_correlation_heatmap,
    plot_seaborn_heatmap,
    plot_scatter,
    get_numeric_summary_table,
    get_categorical_frequency_table,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Data Cleaning & Visualization",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for modern UI
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 14px 18px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetric"]:hover {
        border-color: #93c5fd;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.07);
    }
    /* Headers */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    /* Badge tags */
    .badge-clean {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-raw {
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        padding: 8px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = None
if "cleaned_df" not in st.session_state:
    st.session_state["cleaned_df"] = None
if "audit_log" not in st.session_state:
    st.session_state["audit_log"] = []
if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = "Sample HR Dataset"
if "cleaning_done" not in st.session_state:
    st.session_state["cleaning_done"] = False


# Helper to load sample dataset
@st.cache_data
def load_sample_dataset() -> pd.DataFrame:
    path = os.path.join(os.path.dirname(__file__), "sample_data", "sample_dataset.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


# -----------------------------------------------------------------------------
# Sidebar: Data Source & Cleaning Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.caption("Upload raw CSV or use our pre-built sample dataset.")

    # Data Source Selection
    source_choice = st.radio(
        "Choose Data Source:",
        options=["📊 Sample Dataset (HR Analytics)", "📁 Upload Your Own CSV"],
        index=0,
    )

    if source_choice == "📊 Sample Dataset (HR Analytics)":
        if st.session_state["raw_df"] is None or st.session_state["dataset_name"] != "Sample HR Dataset":
            sample_df = load_sample_dataset()
            st.session_state["raw_df"] = sample_df.copy()
            st.session_state["cleaned_df"] = sample_df.copy()
            st.session_state["dataset_name"] = "Sample HR Dataset"
            st.session_state["cleaning_done"] = False
            st.session_state["audit_log"] = []
        st.success("Loaded **Sample HR Analytics** dataset (with intentional flaws for demonstration).")
    else:
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=["csv"],
            help="Select a CSV file from your computer.",
        )
        if uploaded_file is not None:
            try:
                user_df = pd.read_csv(uploaded_file)
                if st.session_state["dataset_name"] != uploaded_file.name:
                    st.session_state["raw_df"] = user_df.copy()
                    st.session_state["cleaned_df"] = user_df.copy()
                    st.session_state["dataset_name"] = uploaded_file.name
                    st.session_state["cleaning_done"] = False
                    st.session_state["audit_log"] = []
                st.success(f"Loaded **{uploaded_file.name}** successfully!")
            except Exception as e:
                st.error(f"Error parsing CSV file: {e}")

    st.markdown("---")
    st.markdown("### 🧹 Cleaning Options")

    # Cleaning controls
    remove_duplicates_flag = st.checkbox(
        "Remove Duplicate Rows",
        value=True,
        help="Find and delete identical duplicate records.",
    )

    st.markdown("**Missing Value Handling**")
    numeric_strategy = st.selectbox(
        "Numeric Columns Strategy",
        options=["mean", "median", "drop_rows", "none"],
        index=0,
        format_func=lambda x: {
            "mean": "Fill with Mean (Average)",
            "median": "Fill with Median",
            "drop_rows": "Drop Rows with Missing Values",
            "none": "Keep Missing (Do Nothing)",
        }[x],
    )

    categorical_strategy = st.selectbox(
        "Categorical Columns Strategy",
        options=["mode", "custom_value", "drop_rows", "none"],
        index=0,
        format_func=lambda x: {
            "mode": "Fill with Mode (Most Frequent)",
            "custom_value": "Fill with Custom Value",
            "drop_rows": "Drop Rows with Missing Values",
            "none": "Keep Missing (Do Nothing)",
        }[x],
    )

    custom_cat_fill = "Unknown"
    if categorical_strategy == "custom_value":
        custom_cat_fill = st.text_input("Custom Replacement Value", value="Unknown")

    st.markdown("**Outlier Handling (IQR Method)**")
    filter_outliers_flag = st.checkbox(
        "Detect & Remove Outliers",
        value=True,
        help="Use Interquartile Range (Q1 - 1.5*IQR, Q3 + 1.5*IQR) to filter extreme outliers.",
    )

    outlier_cols_selected = []
    iqr_multiplier = 1.5

    if filter_outliers_flag and st.session_state["raw_df"] is not None:
        raw_num_cols = st.session_state["raw_df"].select_dtypes(include=[np.number]).columns.tolist()
        outlier_cols_selected = st.multiselect(
            "Select Columns for Outlier Detection",
            options=raw_num_cols,
            default=raw_num_cols,
            help="IQR will be applied only to these selected columns.",
        )
        iqr_multiplier = st.slider(
            "IQR Multiplier Threshold",
            min_value=1.0,
            max_value=3.0,
            value=1.5,
            step=0.1,
            help="Standard is 1.5. Higher values are more lenient; lower values are stricter.",
        )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        apply_btn = st.button("✨ Apply Cleaning", type="primary", use_container_width=True)
    with col_btn2:
        reset_btn = st.button("↺ Reset", use_container_width=True)

    if reset_btn and st.session_state["raw_df"] is not None:
        st.session_state["cleaned_df"] = st.session_state["raw_df"].copy()
        st.session_state["cleaning_done"] = False
        st.session_state["audit_log"] = []
        st.rerun()

    if apply_btn and st.session_state["raw_df"] is not None:
        cleaned_res, audit_trail, _, _ = clean_dataset(
            st.session_state["raw_df"],
            remove_dups=remove_duplicates_flag,
            numeric_strategy=numeric_strategy,
            categorical_strategy=categorical_strategy,
            custom_cat_fill=custom_cat_fill,
            filter_outliers=filter_outliers_flag,
            outlier_cols=outlier_cols_selected if outlier_cols_selected else None,
            iqr_multiplier=iqr_multiplier,
        )
        st.session_state["cleaned_df"] = cleaned_res
        st.session_state["audit_log"] = audit_trail
        st.session_state["cleaning_done"] = True
        st.success("Cleaning pipeline executed successfully!")
        st.rerun()


# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown('<h1 class="main-title">🧹 Data Cleaning & Visualization</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">An end-to-end interactive platform to inspect data health, execute robust cleaning pipelines, explore interactive charts, and discover actionable findings.</p>',
    unsafe_allow_html=True,
)

# Safety check if DataFrame is available
if st.session_state["raw_df"] is None or st.session_state["raw_df"].empty:
    st.info("👋 Please select the sample dataset or upload a CSV file from the left sidebar to get started.")
    st.stop()

raw_df = st.session_state["raw_df"]
cleaned_df = st.session_state["cleaned_df"] if st.session_state["cleaned_df"] is not None else raw_df
raw_stats = get_dataset_summary(raw_df)
clean_stats = get_dataset_summary(cleaned_df)

# Top Banner Status
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown(f"**Active Dataset:** `{st.session_state['dataset_name']}`")
with col_h2:
    if st.session_state["cleaning_done"]:
        st.markdown('<span class="badge-clean">✓ Cleaned Dataset Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-raw">⚠ Raw Dataset Active (Uncleaned)</span>', unsafe_allow_html=True)

st.write("")

# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tab_overview, tab_cleaning, tab_visuals, tab_insights = st.tabs([
    "📋 Raw Data & Quality Audit",
    "🧹 Data Cleaning & Comparison",
    "📊 Interactive Visualizations",
    "💡 Key Findings & Insights",
])


# =============================================================================
# TAB 1: Raw Data & Quality Audit
# =============================================================================
with tab_overview:
    st.subheader("1. Original Dataset Overview & Quality Profile")
    st.caption("Inspect the initial health, missing values, duplicates, and column structure.")

    # High level KPI cards
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Rows", f"{raw_stats['total_rows']:,}")
    m2.metric("Total Columns", f"{raw_stats['total_columns']}")
    m3.metric(
        "Missing Values",
        f"{raw_stats['total_missing']:,}",
        delta=f"{raw_stats['missing_percentage']}% of cells",
        delta_color="inverse",
    )
    m4.metric(
        "Duplicate Rows",
        f"{raw_stats['duplicate_rows']:,}",
        delta=f"{raw_stats['duplicate_percentage']}% of rows",
        delta_color="inverse",
    )
    m5.metric("Memory Usage", f"{raw_stats['memory_usage_kb']} KB")

    st.markdown("---")

    # Data preview
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.markdown("#### 📄 Dataset Preview")
    with col_t2:
        num_preview_rows = st.slider("Preview Rows", 5, 50, 10, key="raw_preview_slider")

    st.dataframe(raw_df.head(num_preview_rows), use_container_width=True)

    # Breakdown sections in 2 columns
    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("#### ❓ Missing Values Analysis")
        missing_table = get_missing_value_table(raw_df)
        cols_with_missing = missing_table[missing_table["Missing Count"] > 0]
        if not cols_with_missing.empty:
            st.dataframe(cols_with_missing, use_container_width=True)
        else:
            st.success("🎉 No missing values detected in the raw dataset!")

    with c_right:
        st.markdown("#### 👥 Duplicate Records Analysis")
        dup_rows = get_duplicate_rows(raw_df)
        if not dup_rows.empty:
            st.warning(f"Detected **{len(dup_rows)} duplicate row(s)**:")
            st.dataframe(dup_rows.head(10), use_container_width=True)
        else:
            st.success("🎉 No duplicate rows found in the raw dataset!")


# =============================================================================
# TAB 2: Data Cleaning & Comparison
# =============================================================================
with tab_cleaning:
    st.subheader("2. Data Cleaning Pipeline & Audit Trail")
    st.caption("Review before-and-after statistics, outlier boundaries, and download the cleaned data.")

    # Side-by-side Before vs After metrics
    st.markdown("#### 📊 Before vs. After Cleaning Statistics")
    c_b1, c_b2, c_b3, c_b4 = st.columns(4)

    delta_rows = clean_stats["total_rows"] - raw_stats["total_rows"]
    delta_missing = clean_stats["total_missing"] - raw_stats["total_missing"]
    delta_dups = clean_stats["duplicate_rows"] - raw_stats["duplicate_rows"]
    delta_mem = round(clean_stats["memory_usage_kb"] - raw_stats["memory_usage_kb"], 2)

    c_b1.metric("Rows", f"{clean_stats['total_rows']:,}", delta=f"{delta_rows} rows")
    c_b2.metric("Missing Cells", f"{clean_stats['total_missing']:,}", delta=f"{delta_missing}", delta_color="inverse")
    c_b3.metric("Duplicate Rows", f"{clean_stats['duplicate_rows']:,}", delta=f"{delta_dups}", delta_color="inverse")
    c_b4.metric("Memory Footprint", f"{clean_stats['memory_usage_kb']} KB", delta=f"{delta_mem} KB")

    st.markdown("---")

    # Cleaning Audit Log
    st.markdown("#### 📝 Cleaning Audit Trail")
    if st.session_state["audit_log"]:
        for step_idx, step_msg in enumerate(st.session_state["audit_log"], start=1):
            st.markdown(f"**Step {step_idx}:** {step_msg}")
    else:
        st.info("Cleaning pipeline has not been applied yet. Click **'Apply Cleaning'** in the sidebar.")

    # Outlier IQR inspection
    st.markdown("#### 📦 Outlier Detection Details (IQR Method)")
    raw_num_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
    if raw_num_cols:
        outlier_summary_df, _ = detect_outliers_iqr(raw_df, numeric_cols=raw_num_cols, multiplier=1.5)
        st.dataframe(outlier_summary_df, use_container_width=True)
    else:
        st.info("No numeric columns available for outlier calculation.")

    st.markdown("---")

    # Cleaned DataFrame preview
    st.markdown("#### ✅ Cleaned Dataset Preview")
    st.dataframe(cleaned_df.head(15), use_container_width=True)

    # Download Button
    csv_bytes = cleaned_df.to_csv(index=False).encode("utf-8")
    download_filename = f"cleaned_{st.session_state['dataset_name']}"
    if not download_filename.endswith(".csv"):
        download_filename += ".csv"

    st.download_button(
        label="📥 Download Cleaned CSV File",
        data=csv_bytes,
        file_name=download_filename,
        mime="text/csv",
        type="primary",
        help="Export the cleaned dataset to your computer.",
    )


# =============================================================================
# TAB 3: Interactive Visualizations
# =============================================================================
with tab_visuals:
    st.subheader("3. Interactive Exploratory Visualizations")
    st.caption("Generate dynamic charts tailored to column data types.")

    # Choose dataset to visualize: cleaned vs raw
    df_to_visualize = cleaned_df if st.session_state["cleaning_done"] else raw_df
    visual_mode = st.radio(
        "Plot Data From:",
        options=["Cleaned Dataset", "Raw Dataset"],
        horizontal=True,
        index=0 if st.session_state["cleaning_done"] else 1,
    )
    selected_df = cleaned_df if visual_mode == "Cleaned Dataset" else raw_df

    num_cols = selected_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = selected_df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    chart_type = st.selectbox(
        "Select Chart Type:",
        options=[
            "📈 Numeric Distribution (Histogram & Box)",
            "📊 Categorical Distribution (Bar Chart)",
            "📦 Outlier Analysis (Box Plots)",
            "🔗 Correlation Heatmap (Matrix)",
            "🎯 Bivariate Scatter Plot",
            "📋 Descriptive Statistics Tables",
        ],
    )

    st.markdown("---")

    # 1. Numeric Histogram
    if chart_type == "📈 Numeric Distribution (Histogram & Box)":
        if not num_cols:
            st.warning("⚠️ No numeric columns found in the active dataset.")
        else:
            col_sel, bin_sel = st.columns([2, 1])
            with col_sel:
                hist_col = st.selectbox("Choose Numeric Column", options=num_cols)
            with bin_sel:
                nbins = st.slider("Number of Bins", 10, 60, 25)

            if hist_col:
                fig = plot_numeric_histogram(selected_df, hist_col, nbins=nbins)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

    # 2. Categorical Bar
    elif chart_type == "📊 Categorical Distribution (Bar Chart)":
        if not cat_cols:
            st.warning("⚠️ No categorical columns found in the active dataset.")
        else:
            c1, c2 = st.columns([2, 1])
            with c1:
                bar_col = st.selectbox("Choose Categorical Column", options=cat_cols)
            with c2:
                top_n = st.slider("Max Categories to Show", 3, 20, 10)

            if bar_col:
                fig = plot_categorical_bar(selected_df, bar_col, top_n=top_n)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown(f"##### Frequency Table for `{bar_col}`")
                freq_table = get_categorical_frequency_table(selected_df, bar_col)
                st.dataframe(freq_table, use_container_width=True)

    # 3. Box Plots
    elif chart_type == "📦 Outlier Analysis (Box Plots)":
        if not num_cols:
            st.warning("⚠️ No numeric columns found in the active dataset.")
        else:
            box_cols = st.multiselect(
                "Choose Numeric Columns for Box Plot",
                options=num_cols,
                default=num_cols[: min(4, len(num_cols))],
            )
            if box_cols:
                fig = plot_box_plots(selected_df, box_cols)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select at least one numeric column above.")

    # 4. Correlation Heatmap
    elif chart_type == "🔗 Correlation Heatmap (Matrix)":
        if len(num_cols) < 2:
            st.warning("⚠️ At least 2 numeric columns are required to compute correlations.")
        else:
            heatmap_cols = st.multiselect(
                "Choose Numeric Columns to Correlate",
                options=num_cols,
                default=num_cols,
            )
            engine = st.radio("Rendering Engine", ["Plotly (Interactive)", "Seaborn / Matplotlib (Publication Style)"], horizontal=True)

            if len(heatmap_cols) >= 2:
                if engine.startswith("Plotly"):
                    fig = plot_correlation_heatmap(selected_df, cols=heatmap_cols)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    sns_fig = plot_seaborn_heatmap(selected_df, cols=heatmap_cols)
                    if sns_fig:
                        st.pyplot(sns_fig)
            else:
                st.info("Please select at least 2 columns to calculate correlation.")

    # 5. Scatter Plot
    elif chart_type == "🎯 Bivariate Scatter Plot":
        if len(num_cols) < 2:
            st.warning("⚠️ At least 2 numeric columns are required to plot relationships.")
        else:
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                x_col = st.selectbox("X-Axis (Numeric)", options=num_cols, index=0)
            with sc2:
                y_col = st.selectbox("Y-Axis (Numeric)", options=num_cols, index=min(1, len(num_cols) - 1))
            with sc3:
                color_options = ["None"] + cat_cols
                hue_col = st.selectbox("Color / Group By (Optional)", options=color_options)

            trend_flag = st.checkbox("Show Linear Trendline (OLS)", value=True)

            if x_col and y_col:
                fig = plot_scatter(selected_df, x_col, y_col, color_col=hue_col, trendline=trend_flag)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

    # 6. Summary Tables
    elif chart_type == "📋 Descriptive Statistics Tables":
        st.markdown("#### 📐 Numeric Summary Statistics")
        num_summary = get_numeric_summary_table(selected_df)
        if not num_summary.empty:
            st.dataframe(num_summary, use_container_width=True)
        else:
            st.info("No numeric columns present.")

        st.markdown("#### 🏷️ Categorical Summary Overview")
        if cat_cols:
            selected_cat = st.selectbox("Select Categorical Column for Breakdown", options=cat_cols)
            if selected_cat:
                freq = get_categorical_frequency_table(selected_df, selected_cat)
                st.dataframe(freq, use_container_width=True)
        else:
            st.info("No categorical columns present.")


# =============================================================================
# TAB 4: Key Findings & Insights
# =============================================================================
with tab_insights:
    st.subheader("4. Automated Key Findings & Executive Insights")
    st.caption("Data-backed insights dynamically derived from the active dataset.")

    analysis_df = cleaned_df if st.session_state["cleaning_done"] else raw_df
    num_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = analysis_df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    # Section 1: Data Hygiene Summary
    st.markdown("### 🛡️ 1. Data Quality & Preprocessing Summary")
    records_analyzed = len(raw_df)
    missing_handled = raw_stats["total_missing"] - clean_stats["total_missing"]
    dups_removed = raw_stats["duplicate_rows"] - clean_stats["duplicate_rows"]
    rows_retained = len(cleaned_df)
    retention_pct = round((rows_retained / records_analyzed * 100), 1) if records_analyzed > 0 else 0.0

    st.markdown(
        f"""
        - **Total Records Analyzed:** `{records_analyzed}` rows originally ingested.
        - **Missing Values Addressed:** `{missing_handled}` missing data points handled via chosen imputation/drop strategies.
        - **Duplicate Rows Removed:** `{dups_removed}` redundant duplicate rows purged.
        - **Data Retention Rate:** `{retention_pct}%` ({rows_retained} clean rows retained for high-fidelity modeling).
        """
    )

    # Section 2: Outlier Insights
    st.markdown("### 📦 2. Outlier Observations")
    if num_cols:
        outlier_summary_df, _ = detect_outliers_iqr(raw_df, numeric_cols=num_cols, multiplier=1.5)
        top_outliers = outlier_summary_df.sort_values(by="Outliers", ascending=False)
        highest_outlier_col = top_outliers.iloc[0] if not top_outliers.empty else None

        if highest_outlier_col is not None and highest_outlier_col["Outliers"] > 0:
            st.markdown(
                f"- **Primary Outlier Feature:** `{highest_outlier_col['Column']}` exhibited the highest anomaly count "
                f"with **{highest_outlier_col['Outliers']} outlier(s)** "
                f"({highest_outlier_col['Outlier (%)']}% of records outside "
                f"[{highest_outlier_col['Lower Bound']:,}, {highest_outlier_col['Upper Bound']:,}])."
            )
        else:
            st.markdown("- **Outlier Profile:** No significant outliers found based on the standard 1.5x IQR boundary.")
    else:
        st.markdown("- No numeric columns to evaluate for outliers.")

    # Section 3: Correlation & Relationship Patterns
    st.markdown("### 🔗 3. Correlation & Relationship Patterns")
    if len(num_cols) >= 2:
        corr_matrix = analysis_df[num_cols].corr()
        np.fill_diagonal(corr_matrix.values, np.nan)
        unstacked_corr = corr_matrix.unstack().dropna()

        if not unstacked_corr.empty:
            max_pair = unstacked_corr.idxmax()
            max_val = round(float(unstacked_corr.max()), 3)
            min_pair = unstacked_corr.idxmin()
            min_val = round(float(unstacked_corr.min()), 3)

            st.markdown(
                f"- **Strongest Positive Correlation:** Between `{max_pair[0]}` and `{max_pair[1]}` "
                f"(*r* = **{max_val}**). Higher values in `{max_pair[0]}` correspond with higher values in `{max_pair[1]}`."
            )
            if min_val < 0:
                st.markdown(
                    f"- **Strongest Negative / Inverse Correlation:** Between `{min_pair[0]}` and `{min_pair[1]}` "
                    f"(*r* = **{min_val}**)."
                )
            else:
                st.markdown(
                    f"- **Lowest Correlation:** Between `{min_pair[0]}` and `{min_pair[1]}` (*r* = **{min_val}**)."
                )
    else:
        st.markdown("- Not enough numeric variables to assess pairwise correlations.")

    # Section 4: Categorical Dominance
    st.markdown("### 📊 4. Categorical Insights")
    if cat_cols:
        for cat in cat_cols[:3]:  # Top 3 categorical columns
            top_val = analysis_df[cat].mode(dropna=True)
            if not top_val.empty:
                val = top_val.iloc[0]
                cnt = (analysis_df[cat] == val).sum()
                pct = round(cnt / len(analysis_df) * 100, 1) if len(analysis_df) > 0 else 0
                st.markdown(
                    f"- In column `{cat}`, the dominant category is **'{val}'** with **{cnt} entries** ({pct}% share)."
                )
    else:
        st.markdown("- No categorical columns present for category distribution analysis.")

    st.markdown("---")

    # Faculty Demonstration Summary
    with st.expander("🎓 Faculty Demonstration & Viva Guide", expanded=False):
        st.markdown(
            """
            ### How to present this project during review:
            1. **Introduction:** State the objective: building an interactive web-based data cleaning and visualization pipeline to handle messy real-world datasets.
            2. **Data Ingestion:** Showcase both the curated sample dataset and the ability to upload any arbitrary CSV.
            3. **Data Quality Profiling:** Show Tab 1 to demonstrate how missing values and duplicates are identified before touching the data.
            4. **Cleaning Pipeline:** Show Tab 2 to demonstrate:
               - Handling missing values using statistical measures (mean, median, mode) without data leakage.
               - Removal of duplicate records.
               - Interquartile Range (IQR) detection: $IQR = Q_3 - Q_1$, filtering anomalies beyond boundaries.
               - Before & After metrics proving data cleanliness.
               - Exporting the verified cleaned dataset as CSV.
            5. **Visualizations:** Showcase Tab 3 with dynamic distributions, heatmaps, and scatter plots.
            6. **Key Findings:** Conclude with Tab 4, demonstrating that all insights are dynamically derived from real dataset computation.
            """
        )
