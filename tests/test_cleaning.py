"""
Unit tests for data_cleaning module and edge case validation.
"""

import unittest
import pandas as pd
import numpy as np
from src.data_cleaning import (
    get_dataset_summary,
    get_missing_value_table,
    get_duplicate_rows,
    remove_duplicates,
    handle_missing_values,
    detect_outliers_iqr,
    remove_outliers,
    clean_dataset,
)


class TestDataCleaning(unittest.TestCase):

    def setUp(self):
        """Create a standard test dataset with known issues."""
        self.df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 5],
            "age": [20.0, 30.0, np.nan, 40.0, 25.0, 25.0],
            "salary": [50000.0, 60000.0, 70000.0, 1000000.0, 55000.0, 55000.0],
            "dept": ["HR", "IT", "HR", "Sales", None, None],
        })

    def test_get_dataset_summary(self):
        summary = get_dataset_summary(self.df)
        self.assertEqual(summary["total_rows"], 6)
        self.assertEqual(summary["total_columns"], 4)
        self.assertEqual(summary["duplicate_rows"], 1)
        self.assertGreater(summary["total_missing"], 0)
        self.assertIn("salary", summary["numeric_columns"])
        self.assertIn("dept", summary["categorical_columns"])

    def test_get_missing_value_table(self):
        table = get_missing_value_table(self.df)
        self.assertIsInstance(table, pd.DataFrame)
        self.assertIn("Missing Count", table.columns)
        dept_row = table[table["Column"] == "dept"].iloc[0]
        self.assertEqual(dept_row["Missing Count"], 2)

    def test_duplicate_removal(self):
        dups = get_duplicate_rows(self.df)
        self.assertEqual(len(dups), 1)

        cleaned, removed = remove_duplicates(self.df)
        self.assertEqual(removed, 1)
        self.assertEqual(len(cleaned), 5)
        # Ensure original dataframe was not modified
        self.assertEqual(len(self.df), 6)

    def test_missing_values_mean_and_mode(self):
        cleaned, log = handle_missing_values(
            self.df,
            numeric_strategy="mean",
            categorical_strategy="mode",
        )
        # Age had mean (20+30+40+25+25)/5 = 28.0
        self.assertEqual(cleaned["age"].isna().sum(), 0)
        self.assertEqual(cleaned["age"].iloc[2], 28.0)
        # Dept mode is 'HR'
        self.assertEqual(cleaned["dept"].isna().sum(), 0)
        self.assertEqual(cleaned["dept"].iloc[4], "HR")
        self.assertTrue(len(log) > 0)

    def test_missing_values_median_and_custom(self):
        cleaned, log = handle_missing_values(
            self.df,
            numeric_strategy="median",
            categorical_strategy="custom_value",
            custom_cat_fill="Unassigned",
        )
        self.assertEqual(cleaned["age"].isna().sum(), 0)
        self.assertEqual(cleaned["dept"].iloc[4], "Unassigned")

    def test_missing_values_drop_rows(self):
        cleaned, log = handle_missing_values(
            self.df,
            numeric_strategy="drop_rows",
            categorical_strategy="drop_rows",
        )
        self.assertEqual(cleaned.isna().sum().sum(), 0)
        self.assertLess(len(cleaned), len(self.df))

    def test_outlier_detection_iqr(self):
        summary_df, mask = detect_outliers_iqr(self.df, numeric_cols=["salary"], multiplier=1.5)
        self.assertEqual(len(summary_df), 1)
        # Salary 1,000,000 should be detected as outlier
        self.assertTrue(mask.iloc[3])
        self.assertEqual(summary_df.iloc[0]["Outliers"], 1)

    def test_outlier_removal(self):
        cleaned, count, summary = remove_outliers(self.df, numeric_cols=["salary"], multiplier=1.5)
        self.assertEqual(count, 1)
        self.assertEqual(len(cleaned), 5)
        self.assertNotIn(1000000.0, cleaned["salary"].values)

    def test_clean_dataset_pipeline(self):
        cleaned, log, before, after = clean_dataset(
            self.df,
            remove_dups=True,
            numeric_strategy="median",
            categorical_strategy="mode",
            filter_outliers=True,
            outlier_cols=["salary"],
        )
        # 6 original -> minus 1 dup = 5 -> minus 1 outlier = 4 rows
        self.assertEqual(len(cleaned), 4)
        self.assertEqual(after["total_missing"], 0)
        self.assertEqual(after["duplicate_rows"], 0)
        # Original df remains untouched
        self.assertEqual(len(self.df), 6)

    def test_edge_cases_empty_df(self):
        empty_df = pd.DataFrame()
        summary = get_dataset_summary(empty_df)
        self.assertEqual(summary["total_rows"], 0)

        cleaned, count = remove_duplicates(empty_df)
        self.assertEqual(len(cleaned), 0)

        cleaned, log = handle_missing_values(empty_df)
        self.assertEqual(len(cleaned), 0)

        summary_df, mask = detect_outliers_iqr(empty_df)
        self.assertEqual(len(summary_df), 0)

    def test_edge_case_all_nan_column(self):
        all_nan_df = pd.DataFrame({
            "a": [np.nan, np.nan, np.nan],
            "b": ["x", "y", "z"],
        })
        cleaned, log = handle_missing_values(all_nan_df, numeric_strategy="mean")
        self.assertEqual(cleaned["a"].isna().sum(), 0)
        self.assertEqual(cleaned["a"].iloc[0], 0.0)


if __name__ == "__main__":
    unittest.main()
