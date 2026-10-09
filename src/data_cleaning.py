"""
Data Cleaning Module
--------------------
Provides pure Pandas and NumPy functions for inspecting, profiling,
and cleaning tabular datasets:
- Missing value analysis & imputation (mean, median, mode, custom, drop)
- Duplicate detection and removal
- Outlier detection and filtering using the Interquartile Range (IQR) method
- Before-and-after statistical profiling and audit tracking
"""

from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes top-level metadata and hygiene statistics for a DataFrame.

    Returns:
        dict: Summary statistics including rows, columns, missing count,
              duplicate count, and memory usage.
    """
    if df is None or df.empty:
        return {
            "total_rows": 0,
            "total_columns": 0,
            "total_missing": 0,
            "missing_percentage": 0.0,
            "duplicate_rows": 0,
            "duplicate_percentage": 0.0,
            "numeric_columns": [],
            "categorical_columns": [],
            "memory_usage_kb": 0.0,
        }

    total_rows = len(df)
    total_cells = df.size
    total_missing = int(df.isna().sum().sum())
    missing_pct = round((total_missing / total_cells * 100), 2) if total_cells > 0 else 0.0

    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = round((duplicate_rows / total_rows * 100), 2) if total_rows > 0 else 0.0

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    # Memory in Kilobytes
    memory_kb = round(df.memory_usage(deep=True).sum() / 1024, 2)

    return {
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "total_missing": total_missing,
        "missing_percentage": missing_pct,
        "duplicate_rows": duplicate_rows,
        "duplicate_percentage": duplicate_pct,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "memory_usage_kb": memory_kb,
    }


def get_missing_value_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a breakdown of missing values per column.

    Returns:
        pd.DataFrame: Table with Missing Count, Missing %, and Data Type.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Column", "Missing Count", "Missing (%)", "Data Type"])

    missing_counts = df.isna().sum()
    total_rows = len(df)
    missing_pct = (missing_counts / total_rows * 100).round(2) if total_rows > 0 else 0.0
    dtypes = df.dtypes.astype(str)

    report = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": missing_counts.values,
        "Missing (%)": missing_pct.values,
        "Data Type": dtypes.values,
    })

    return report.sort_values(by="Missing Count", ascending=False).reset_index(drop=True)


def get_duplicate_rows(df: pd.DataFrame, keep: str = "first") -> pd.DataFrame:
    """
    Returns all rows identified as duplicates.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    return df[df.duplicated(keep=keep)].copy()


def remove_duplicates(df: pd.DataFrame, keep: str = "first") -> Tuple[pd.DataFrame, int]:
    """
    Removes duplicate records from the DataFrame.

    Returns:
        Tuple[pd.DataFrame, int]: Cleaned DataFrame and count of removed rows.
    """
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame(), 0

    initial_count = len(df)
    cleaned_df = df.drop_duplicates(keep=keep).reset_index(drop=True)
    removed_count = initial_count - len(cleaned_df)
    return cleaned_df, removed_count


def handle_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = "mean",
    categorical_strategy: str = "mode",
    custom_cat_fill: str = "Missing",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Applies imputation or row dropping strategies for missing values.

    Args:
        df: Input DataFrame.
        numeric_strategy: 'mean', 'median', 'drop_rows', or 'none'.
        categorical_strategy: 'mode', 'custom_value', 'drop_rows', or 'none'.
        custom_cat_fill: String value used when categorical_strategy == 'custom_value'.

    Returns:
        Tuple[pd.DataFrame, List[str]]: Cleaned DataFrame and an audit trail log.
    """
    if df is None or df.empty:
        return (df.copy() if df is not None else pd.DataFrame()), ["Dataset is empty; no missing value operations performed."]

    cleaned_df = df.copy()
    audit_log: List[str] = []

    # If user selected drop_rows for either or both:
    if numeric_strategy == "drop_rows" and categorical_strategy == "drop_rows":
        before = len(cleaned_df)
        cleaned_df = cleaned_df.dropna().reset_index(drop=True)
        dropped = before - len(cleaned_df)
        audit_log.append(f"Dropped {dropped} rows containing missing values across all columns.")
        return cleaned_df, audit_log

    # Handle numeric columns
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_strategy == "drop_rows":
        before = len(cleaned_df)
        cleaned_df = cleaned_df.dropna(subset=numeric_cols).reset_index(drop=True)
        dropped = before - len(cleaned_df)
        audit_log.append(f"Dropped {dropped} rows containing missing values in numeric columns.")
    elif numeric_strategy in ("mean", "median"):
        for col in numeric_cols:
            missing_count = cleaned_df[col].isna().sum()
            if missing_count > 0:
                if cleaned_df[col].dropna().empty:
                    # All values in column were NaN, fill with 0
                    fill_val = 0.0
                    cleaned_df[col] = cleaned_df[col].fillna(fill_val)
                    audit_log.append(f"Column '{col}' had all NaN values; filled {missing_count} cells with 0.")
                else:
                    fill_val = cleaned_df[col].mean() if numeric_strategy == "mean" else cleaned_df[col].median()
                    fill_val = round(fill_val, 2)
                    cleaned_df[col] = cleaned_df[col].fillna(fill_val)
                    audit_log.append(
                        f"Imputed {missing_count} missing values in '{col}' using {numeric_strategy} ({fill_val})."
                    )

    # Handle categorical / object columns
    cat_cols = cleaned_df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    if categorical_strategy == "drop_rows":
        before = len(cleaned_df)
        cleaned_df = cleaned_df.dropna(subset=cat_cols).reset_index(drop=True)
        dropped = before - len(cleaned_df)
        audit_log.append(f"Dropped {dropped} rows containing missing values in categorical columns.")
    elif categorical_strategy == "mode":
        for col in cat_cols:
            missing_count = cleaned_df[col].isna().sum()
            if missing_count > 0:
                mode_series = cleaned_df[col].mode(dropna=True)
                fill_val = mode_series.iloc[0] if not mode_series.empty else "Unknown"
                cleaned_df[col] = cleaned_df[col].fillna(fill_val)
                audit_log.append(
                    f"Imputed {missing_count} missing values in '{col}' using mode ('{fill_val}')."
                )
    elif categorical_strategy == "custom_value":
        fill_val = str(custom_cat_fill).strip() if str(custom_cat_fill).strip() else "Unknown"
        for col in cat_cols:
            missing_count = cleaned_df[col].isna().sum()
            if missing_count > 0:
                cleaned_df[col] = cleaned_df[col].fillna(fill_val)
                audit_log.append(
                    f"Imputed {missing_count} missing values in '{col}' with custom value ('{fill_val}')."
                )

    if not audit_log:
        audit_log.append("No missing values found or strategy set to 'none'.")

    return cleaned_df, audit_log


def detect_outliers_iqr(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    multiplier: float = 1.5,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Detects outliers across numeric columns using the Interquartile Range (IQR) method.

    Rule:
        IQR = Q3 - Q1
        Lower Bound = Q1 - multiplier * IQR
        Upper Bound = Q3 + multiplier * IQR
        Outlier if value < Lower Bound or value > Upper Bound

    Returns:
        Tuple[pd.DataFrame, pd.Series]:
            - Summary DataFrame with Q1, Q3, IQR, bounds, and outlier count per column.
            - Boolean Series of length len(df) indicating which rows have outliers.
    """
    if df is None or df.empty:
        empty_summary = pd.DataFrame(
            columns=["Column", "Q1 (25%)", "Q3 (75%)", "IQR", "Lower Bound", "Upper Bound", "Outliers", "Outlier (%)"]
        )
        return empty_summary, pd.Series(dtype=bool)

    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    summary_rows = []
    overall_mask = pd.Series(False, index=df.index)

    for col in numeric_cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - (multiplier * iqr)
        upper_bound = q3 + (multiplier * iqr)

        # Boolean mask for this column
        col_outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        count = int(col_outlier_mask.sum())
        pct = round((count / len(df) * 100), 2) if len(df) > 0 else 0.0

        overall_mask = overall_mask | col_outlier_mask

        summary_rows.append({
            "Column": col,
            "Q1 (25%)": round(float(q1), 2),
            "Q3 (75%)": round(float(q3), 2),
            "IQR": round(float(iqr), 2),
            "Lower Bound": round(float(lower_bound), 2),
            "Upper Bound": round(float(upper_bound), 2),
            "Outliers": count,
            "Outlier (%)": pct,
        })

    summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame(
        columns=["Column", "Q1 (25%)", "Q3 (75%)", "IQR", "Lower Bound", "Upper Bound", "Outliers", "Outlier (%)"]
    )

    return summary_df, overall_mask


def remove_outliers(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    multiplier: float = 1.5,
) -> Tuple[pd.DataFrame, int, pd.DataFrame]:
    """
    Filters out rows containing IQR outliers in selected numeric columns.

    Returns:
        Tuple[pd.DataFrame, int, pd.DataFrame]:
            - Cleaned DataFrame without outlier rows.
            - Count of removed outlier rows.
            - Outlier summary table.
    """
    if df is None or df.empty:
        return (df.copy() if df is not None else pd.DataFrame()), 0, pd.DataFrame()

    summary_df, outlier_mask = detect_outliers_iqr(df, numeric_cols=numeric_cols, multiplier=multiplier)
    outlier_count = int(outlier_mask.sum())

    cleaned_df = df[~outlier_mask].reset_index(drop=True)
    return cleaned_df, outlier_count, summary_df


def clean_dataset(
    df: pd.DataFrame,
    remove_dups: bool = True,
    numeric_strategy: str = "mean",
    categorical_strategy: str = "mode",
    custom_cat_fill: str = "Unknown",
    filter_outliers: bool = False,
    outlier_cols: Optional[List[str]] = None,
    iqr_multiplier: float = 1.5,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any], Dict[str, Any]]:
    """
    Executes a comprehensive, non-destructive cleaning pipeline.

    Returns:
        Tuple containing:
            - cleaned_df (pd.DataFrame)
            - audit_log (List[str])
            - before_summary (Dict[str, Any])
            - after_summary (Dict[str, Any])
    """
    if df is None or df.empty:
        empty_summary = get_dataset_summary(df)
        return (df.copy() if df is not None else pd.DataFrame()), ["Dataset is empty."], empty_summary, empty_summary

    before_summary = get_dataset_summary(df)
    audit_log: List[str] = []
    working_df = df.copy()

    # Step 1: Remove duplicates
    if remove_dups:
        working_df, dup_count = remove_duplicates(working_df)
        if dup_count > 0:
            audit_log.append(f"Removed {dup_count} duplicate row(s).")
        else:
            audit_log.append("No duplicate rows found.")

    # Step 2: Handle missing values
    working_df, missing_log = handle_missing_values(
        working_df,
        numeric_strategy=numeric_strategy,
        categorical_strategy=categorical_strategy,
        custom_cat_fill=custom_cat_fill,
    )
    audit_log.extend(missing_log)

    # Step 3: Outlier handling
    if filter_outliers:
        working_df, outlier_count, _ = remove_outliers(
            working_df,
            numeric_cols=outlier_cols,
            multiplier=iqr_multiplier,
        )
        if outlier_count > 0:
            audit_log.append(
                f"Removed {outlier_count} outlier row(s) using IQR (multiplier={iqr_multiplier})."
            )
        else:
            audit_log.append("No outliers detected or removed.")

    after_summary = get_dataset_summary(working_df)
    return working_df, audit_log, before_summary, after_summary
