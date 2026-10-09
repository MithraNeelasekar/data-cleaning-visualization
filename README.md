# 🧹 Data Cleaning & Visualization Dashboard

An interactive, production-ready web application built with **Streamlit**, **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**, and **Plotly** to clean messy raw tabular data, perform automated statistical profiling, generate interactive exploratory visualizations, and extract data-driven executive insights.

---

## 📌 1. Project Title & Overview

- **Project Title:** Data Cleaning & Visualization Dashboard
- **Application Type:** Interactive Data Science & Preprocessing Web Application
- **Live Deployed App URL:** `[Paste Your Deployed Streamlit App URL Here]`
- **GitHub Repository:** `[Paste Your GitHub Repository URL Here]`

---

## 🎯 2. Problem Statement & Objectives

### Problem Statement
Real-world datasets collected from business systems, IoT sensors, customer surveys, or transactional databases are rarely clean. They frequently suffer from:
1. **Missing / Null Values:** Incomplete records resulting in biased calculations or model failures.
2. **Duplicate Records:** Artificially inflated sample sizes distorting statistical distributions.
3. **Extreme Outliers:** Data entry anomalies or extreme variances corrupting metrics like mean and standard deviation.
4. **Poor Observability:** Lack of clear before-and-after audit logs to verify what transformations were performed.

### Objectives
- Provide a zero-code web interface where users can upload any CSV dataset or explore a pre-loaded sample dataset.
- Implement mathematically grounded data cleaning techniques (Mean/Median/Mode imputation, Duplicate pruning, Interquartile Range outlier filtering).
- Maintain complete non-destructive data integrity (original raw data is never mutated or overwritten).
- Offer rich, dynamic visualizations (histograms, categorical bar charts, box plots, correlation heatmaps, scatter plots).
- Provide real-time data-driven insights and a downloadable cleaned CSV file for downstream machine learning or reporting.

---

## ✨ 3. Core Features

- **📁 Flexible Data Ingestion:**
  - One-click loading of the curated **Sample HR Analytics Dataset** (pre-configured with intentional nulls, duplicates, and outliers).
  - Drag-and-drop CSV upload for custom user datasets.
- **🛡️ Automated Data Quality Profiling:**
  - KPI metric cards for Total Rows, Columns, Missing Cells, Duplicate Rows, and Memory Footprint.
  - Per-column missing value breakdowns with percentages and data types.
  - Identification and preview of duplicate rows.
- **🧹 Robust Cleaning Pipeline:**
  - **Missing Value Handling:**
    - Numeric columns: Impute with **Mean**, **Median**, or **Drop Rows**.
    - Categorical columns: Impute with **Mode (Most Frequent)**, **Custom Replacement Value**, or **Drop Rows**.
  - **Duplicate Handling:** Exact duplicate detection and removal.
  - **Outlier Handling (IQR Method):** Configurable Interquartile Range bounds ($Q_1 - 1.5 \times IQR$ to $Q_3 + 1.5 \times IQR$) with interactive column selection and multiplier threshold slider.
  - **Audit Trail:** Step-by-step logs recording every cleaning transformation.
- **📊 Comparative Analysis:**
  - Real-time **Before vs. After** statistics with delta counters.
- **📈 Exploratory Data Visualizations:**
  - **Numeric Distributions:** Interactive Plotly histograms with marginal box plots and density curves.
  - **Categorical Distributions:** Frequency-labeled bar charts with percentages and detailed frequency tables.
  - **Outlier Inspection:** Side-by-side interactive Plotly box plots.
  - **Correlation Analysis:** Interactive Plotly correlation matrix and publication-grade Seaborn/Matplotlib heatmaps.
  - **Bivariate Relationships:** Interactive scatter plots with optional color grouping and OLS linear trendlines.
  - **Descriptive Statistics:** Full statistical tables (`mean`, `std`, `min`, quartiles, `max`).
- **💡 Data-Driven Executive Insights:**
  - Dynamic calculations for data retention rates, top outlier drivers, strongest positive/negative correlation coefficients, and category dominance.
- **📥 One-Click Export:**
  - Instant download of the sanitized dataset as a UTF-8 CSV file.

---

## 🛠️ 4. Technology Stack

| Technology | Purpose |
| :--- | :--- |
| **Python 3.10+** | Core programming language |
| **Streamlit** | Web application framework and dashboard interface |
| **Pandas & NumPy** | High-performance data manipulation and statistical math |
| **Plotly Express / Graph Objects** | Interactive, responsive browser-based charts |
| **Matplotlib & Seaborn** | Statistical graphics and publication-style correlation heatmaps |
| **Git & GitHub** | Source code version control and repository management |
| **Streamlit Community Cloud** | Free cloud deployment and public hosting |

---

## 📁 5. Project Directory Structure

```
data-cleaning-visualization/
├── .gitignore                   # Excludes __pycache__, .venv, OS temp files
├── requirements.txt             # Pinned Python package dependencies
├── README.md                    # Project documentation & execution guide
├── app.py                       # Main Streamlit dashboard application
├── sample_data/
│   └── sample_dataset.csv       # Sample dataset with intentional data quality issues
├── src/
│   ├── __init__.py              # Python package marker
│   ├── data_cleaning.py         # Pure Pandas/NumPy cleaning and profiling logic
│   └── visualizations.py        # Chart generation functions (Plotly, Seaborn, Matplotlib)
└── tests/
    ├── __init__.py
    └── test_cleaning.py         # Automated unit tests for data cleaning logic
```

---

## ⚙️ 6. Installation & Local Setup

### Step 1: Open Terminal / PowerShell
Navigate to your project directory:
```powershell
cd "c:\Users\MITHRA N\Desktop\Intenship 1"
```

### Step 2: (Optional but Recommended) Create a Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Run the Application Locally
Launch the Streamlit server:
```powershell
streamlit run app.py
```
*(If `streamlit` is not recognized directly on your PATH, run `python -m streamlit run app.py`)*

The dashboard will open automatically in your browser at:
`http://localhost:8501`

---

## 🧪 7. Running Automated Tests

To run the automated test suite and verify cleaning algorithms:
```powershell
python -m unittest discover -s tests -p "test_*.py"
```

---

## 🔬 8. Detailed Explanation of Data Cleaning Methods

### 1. Handling Missing Values
Missing data can introduce statistical distortion or break machine learning algorithms. The application provides three scientific remedies:
- **Mean Imputation:** Replaces missing numbers with the arithmetic average ($\mu$). Best suited for symmetric, normally distributed numerical data.
- **Median Imputation:** Replaces missing numbers with the 50th percentile. Highly robust against skewed data and outlier contamination.
- **Mode Imputation:** Replaces missing categorical values with the most frequently occurring category.
- **Custom Constant:** Fills categorical nulls with an informative label such as `"Unknown"` or `"Unassigned"`.
- **Row Deletion (`drop_rows`):** Discards records containing missing values. Appropriate when data loss is acceptable and missingness is completely random (MCAR).

### 2. Duplicate Record Removal
Duplicates occur from repeated submissions, ingestion glitches, or system joins. The application uses Pandas' `.drop_duplicates(keep='first')` to retain the initial genuine entry and eliminate identical redundancies.

### 3. Outlier Detection via Interquartile Range (IQR)
The Interquartile Range method is a non-parametric outlier detection technique that does not assume normal distribution:
1. **First Quartile ($Q_1$):** 25th percentile value.
2. **Third Quartile ($Q_3$):** 75th percentile value.
3. **Interquartile Range ($IQR$):**
   $$IQR = Q_3 - Q_1$$
4. **Outlier Boundaries:**
   $$\text{Lower Bound} = Q_1 - (1.5 \times IQR)$$
   $$\text{Upper Bound} = Q_3 + (1.5 \times IQR)$$
5. Any value strictly below the Lower Bound or above the Upper Bound is classified as an anomaly and can be retained or filtered out.

---

## 📊 9. Detailed Explanation of Visualizations

1. **Histogram & Marginal Box Plot:**
   - Visualizes continuous frequency distribution, skewness (left/right tail), and spread.
   - The marginal box plot instantly displays medians and extreme values above the distribution curve.
2. **Categorical Bar Charts:**
   - Highlights category counts with percentage overlays for top categories.
   - Accompanied by full frequency count tables.
3. **Multi-Column Box Plots:**
   - Displays quartile ranges ($Q_1$, Median, $Q_3$, Whiskers) side-by-side for multiple numeric columns.
   - Distinct markers pinpoint individual outlier points.
4. **Correlation Heatmaps:**
   - Computes Pearson correlation coefficients ($r \in [-1, 1]$) between all numeric features.
   - Provides both an interactive Plotly heatmap and a static, high-contrast Seaborn heatmap.
5. **Bivariate Scatter Plots:**
   - Explores interactions between two numeric variables (e.g., *Years of Experience* vs. *Annual Salary*).
   - Allows optional grouping by a categorical column (e.g. *Department*) and includes an Ordinary Least Squares (OLS) linear trendline.

---

## 🚀 10. Publishing the Project to GitHub

Follow these steps to publish your project:

### Step 1: Create a New Repository on GitHub
1. Open your browser and navigate to [github.com/new](https://github.com/new).
2. Set the **Repository name** to: `data-cleaning-visualization`
3. Choose **Public**.
4. **Leave all checkboxes unchecked** (do *not* add README, .gitignore, or license, as we already created them locally).
5. Click **Create repository**.
6. Copy your repository URL (e.g., `https://github.com/<your-username>/data-cleaning-visualization.git`).

### Step 2: Push Local Code via PowerShell
Run the following commands inside `c:\Users\MITHRA N\Desktop\Intenship 1`:

```powershell
# 1. Initialize git (if not already initialized)
git init

# 2. Add all files to staging
git add .

# 3. Create initial commit
git commit -m "feat: complete data cleaning and visualization web dashboard"

# 4. Set default branch to main
git branch -M main

# 5. Connect to your GitHub repository (replace with your actual URL)
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/data-cleaning-visualization.git

# 6. Push to GitHub
git push -u origin main
```

> **Note:** If Git prompts you for authentication:
> - Sign in via GitHub browser prompt or Personal Access Token (PAT).
> - Never share your GitHub password or token in public chats.

---

## ☁️ 11. Deploying on Streamlit Community Cloud

Once pushed to GitHub, deploy the application for free in under 2 minutes:

1. Visit [share.streamlit.io](https://share.streamlit.io/) and click **Sign in with GitHub**.
2. Click **Create app** or **New app**.
3. Fill in the deployment details:
   - **Repository:** `<YOUR_GITHUB_USERNAME>/data-cleaning-visualization`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy!**
5. Streamlit will install dependencies from `requirements.txt` and launch your live web application.
6. Once deployed, copy your live link (e.g., `https://data-cleaning-visualization.streamlit.app`) and paste it at the top of this `README.md`.

---

## 🎓 12. Faculty Demonstration & Viva Walkthrough

When presenting this project to your faculty, professors, or recruiters:

1. **Architecture & Design:**
   - Explain the separation of concerns: `src/data_cleaning.py` handles pure data transformations, `src/visualizations.py` handles plotting logic, and `app.py` renders the responsive Streamlit UI.
2. **Live Demonstration Flow:**
   - **Step 1 (Raw Profile):** Open Tab 1 to showcase the dataset's initial flaws (e.g., missing values in `Age` and `Annual_Salary`, 5 duplicate records, and extreme salary outliers like \$850k and \$920k).
   - **Step 2 (Cleaning Execution):** Open the sidebar, select Imputation strategies (e.g. Mean for numeric, Mode for categorical), enable duplicate removal and IQR outlier detection. Click **"Apply Cleaning"**.
   - **Step 3 (Audit Verification):** Show Tab 2. Point out the Before vs. After KPI delta metrics, the step-by-step audit log, and the IQR boundary table. Click **"Download Cleaned CSV File"** to demonstrate export.
   - **Step 4 (Visual Analytics):** Switch to Tab 3. Showcase the interactive histogram with marginal box, the categorical distribution of departments, and the correlation heatmap demonstrating relationship strengths.
   - **Step 5 (Key Findings):** Review Tab 4 to highlight the dynamically computed data retention rate, correlation metrics, and category dominance.
3. **Viva Questions to Expect:**
   - *Why use IQR over Z-score?* -> IQR is non-parametric and not distorted by extreme outliers, whereas Z-score relies on the mean and standard deviation which are themselves corrupted by outliers.
   - *Why median over mean for skewed data?* -> Median is resistant to extreme values; mean is pulled toward long tails.

---

## 📄 License
This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).
