"""
Visualizations Module
---------------------
Provides plotting utilities for exploratory data analysis:
- Plotly interactive figures (Histograms, Bar charts, Box plots, Scatters, Heatmaps)
- Matplotlib/Seaborn alternative for Correlation Heatmap
- Descriptive summary tables and categorical frequency distributions
- Defensive checks for low column counts, empty data, or missing data types
"""

from typing import List, Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns


# Professional color palette
PRIMARY_COLOR = "#2563EB"   # Modern Indigo / Blue
SECONDARY_COLOR = "#0D9488" # Teal
ACCENT_COLOR = "#F59E0B"    # Amber
BG_TEMPLATE = "plotly_white"


def plot_numeric_histogram(
    df: pd.DataFrame,
    col: str,
    nbins: int = 30,
) -> Optional[go.Figure]:
    """
    Creates an interactive Plotly histogram with a marginal box plot
    to analyze distribution shape, skewness, and outliers.
    """
    if df is None or df.empty or col not in df.columns:
        return None

    clean_series = df[col].dropna()
    if clean_series.empty:
        return None

    fig = px.histogram(
        df,
        x=col,
        nbins=nbins,
        marginal="box",
        title=f"Distribution of {col}",
        color_discrete_sequence=[PRIMARY_COLOR],
        template=BG_TEMPLATE,
    )
    fig.update_layout(
        xaxis_title=col,
        yaxis_title="Count",
        title_font_size=16,
        bargap=0.08,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_categorical_bar(
    df: pd.DataFrame,
    col: str,
    top_n: int = 10,
) -> Optional[go.Figure]:
    """
    Creates an interactive Plotly bar chart showing counts and percentages
    for top categories in a column.
    """
    if df is None or df.empty or col not in df.columns:
        return None

    counts = df[col].astype(str).value_counts().head(top_n).reset_index()
    counts.columns = [col, "Count"]
    total = len(df.dropna(subset=[col])) if not df[col].dropna().empty else len(df)
    counts["Percentage"] = (counts["Count"] / total * 100).round(1)

    fig = px.bar(
        counts,
        x=col,
        y="Count",
        text=counts.apply(lambda r: f"{r['Count']} ({r['Percentage']}%)", axis=1),
        title=f"Top {top_n} Categories in {col}",
        color="Count",
        color_continuous_scale="Blues",
        template=BG_TEMPLATE,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title=col,
        yaxis_title="Frequency",
        title_font_size=16,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_box_plots(
    df: pd.DataFrame,
    cols: List[str],
) -> Optional[go.Figure]:
    """
    Creates side-by-side interactive box plots for one or more numeric columns,
    clearly showing quartiles, medians, and outliers.
    """
    if df is None or df.empty or not cols:
        return None

    valid_cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not valid_cols:
        return None

    fig = go.Figure()
    palette = px.colors.qualitative.Plotly

    for idx, col in enumerate(valid_cols):
        fig.add_trace(go.Box(
            y=df[col].dropna(),
            name=col,
            boxpoints="outliers",
            marker_color=palette[idx % len(palette)],
            line_width=2,
        ))

    fig.update_layout(
        title="Outlier & Quartile Analysis (Box Plots)",
        yaxis_title="Value",
        template=BG_TEMPLATE,
        title_font_size=16,
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=False,
    )
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
) -> Optional[go.Figure]:
    """
    Creates an interactive Plotly correlation heatmap for numeric features.
    """
    if df is None or df.empty:
        return None

    if cols is None:
        numeric_df = df.select_dtypes(include=[np.number])
    else:
        valid_cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        numeric_df = df[valid_cols]

    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr().round(2)

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Pearson Correlation Matrix",
        template=BG_TEMPLATE,
    )
    fig.update_layout(
        title_font_size=16,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_seaborn_heatmap(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
) -> Optional[plt.Figure]:
    """
    Renders a clean static correlation heatmap using Matplotlib and Seaborn.
    """
    if df is None or df.empty:
        return None

    if cols is None:
        numeric_df = df.select_dtypes(include=[np.number])
    else:
        valid_cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        numeric_df = df[valid_cols]

    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap (Seaborn)", fontsize=14, pad=12)
    plt.tight_layout()
    return fig


def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    trendline: bool = True,
) -> Optional[go.Figure]:
    """
    Creates an interactive scatter plot between two numeric variables,
    with an optional categorical color grouping and trendline.
    """
    if df is None or df.empty or x_col not in df.columns or y_col not in df.columns:
        return None

    valid_color = color_col if (color_col and color_col in df.columns and color_col != "None") else None

    # Filter out NaNs in chosen coordinates
    subset_cols = [x_col, y_col] + ([valid_color] if valid_color else [])
    plot_data = df.dropna(subset=subset_cols).copy()
    if plot_data.empty:
        return None

    trend_mode = "ols" if trendline and valid_color is None else None

    fig = px.scatter(
        plot_data,
        x=x_col,
        y=y_col,
        color=valid_color,
        trendline=trend_mode,
        title=f"Relationship: {x_col} vs {y_col}",
        template=BG_TEMPLATE,
        opacity=0.85,
    )
    fig.update_traces(marker=dict(size=9, line=dict(width=1, color="DarkSlateGrey")))
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        title_font_size=16,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def get_numeric_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns descriptive statistics for numeric columns.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return pd.DataFrame()
    return num_df.describe().T.round(2)


def get_categorical_frequency_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Returns value counts and percentages for a categorical column.
    """
    if df is None or df.empty or col not in df.columns:
        return pd.DataFrame(columns=["Value", "Count", "Percentage (%)"])

    counts = df[col].astype(str).value_counts(dropna=False).reset_index()
    counts.columns = ["Value", "Count"]
    total = len(df)
    counts["Percentage (%)"] = (counts["Count"] / total * 100).round(2) if total > 0 else 0.0
    return counts
