"""
Data Preprocessing Module
Handles data loading, cleaning, and quality assessment
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


class DataPreprocessor:
    """
    Complete data preprocessing pipeline for EMI dataset
    """

    def __init__(self, data_path):
        """
        Initialize preprocessor with data path

        Args:
            data_path: Path to raw CSV file
        """
        self.data_path = data_path
        self.df = None
        self.df_clean = None
        self.quality_report = {}

    def load_data(self):
        """Load the dataset and perform initial inspection"""
        print("=" * 60)
        print("STEP 1: LOADING DATA")
        print("=" * 60)

        self.df = pd.read_csv(self.data_path)

        print(f"✓ Dataset loaded successfully")
        print(f"✓ Total Records: {len(self.df):,}")
        print(f"✓ Total Features: {len(self.df.columns)}")
        print(f"\nColumn Names:\n{list(self.df.columns)}")

        return self.df

    def assess_data_quality(self):
        """Comprehensive data quality assessment"""
        print("\n" + "=" * 60)
        print("STEP 2: DATA QUALITY ASSESSMENT")
        print("=" * 60)

        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Missing values analysis
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100

        print("\n1. Missing Values Analysis:")
        if missing.sum() == 0:
            print("✓ No missing values found!")
            missing_df = None
        else:
            missing_df = pd.DataFrame(
                {"Missing_Count": missing[missing > 0], "Percentage": missing_pct[missing > 0]}
            ).sort_values("Percentage", ascending=False)
            print(missing_df)

        self.quality_report["missing_values"] = missing_df

        # Duplicate records
        duplicates = self.df.duplicated().sum()
        print(f"\n2. Duplicate Records: {duplicates:,}")
        self.quality_report["duplicates"] = int(duplicates)

        # Data types
        print("\n3. Data Types:")
        print(self.df.dtypes.value_counts())

        # Basic statistics
        print("\n4. Numerical Features Summary:")
        try:
            print(self.df.describe().T[["mean", "std", "min", "max"]])
        except Exception:
            print("No numerical columns available for summary.")

        # Target variable distribution (guarded)
        print("\n5. Target Variables Distribution:")
        if "emi_eligibility" in self.df.columns:
            print("\nEMI Eligibility (Classification Target):")
            print(self.df["emi_eligibility"].value_counts())
        else:
            print("emi_eligibility column not found.")

        if "max_monthly_emi" in self.df.columns:
            print("\nMax Monthly EMI (Regression Target):")
            print(f"Mean: ₹{self.df['max_monthly_emi'].mean():.2f}")
            print(f"Median: ₹{self.df['max_monthly_emi'].median():.2f}")
            print(
                f"Range: ₹{self.df['max_monthly_emi'].min():.2f} - ₹{self.df['max_monthly_emi'].max():.2f}"
            )
        else:
            print("max_monthly_emi column not found.")

        return self.quality_report

    def handle_missing_values(self):
        """Handle missing values with appropriate strategies"""
        print("\n" + "=" * 60)
        print("STEP 3: HANDLING MISSING VALUES")
        print("=" * 60)

        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        self.df_clean = self.df.copy()

        # Check if there are missing values
        missing_cols = self.df_clean.columns[self.df_clean.isnull().any()].tolist()

        if not missing_cols:
            print("✓ No missing values to handle")
            return self.df_clean

        # Strategy for numerical columns - fill with median
        numerical_cols = self.df_clean.select_dtypes(include=["int64", "float64"]).columns
        for col in numerical_cols:
            if self.df_clean[col].isnull().sum() > 0:
                median_val = self.df_clean[col].median()
                self.df_clean[col].fillna(median_val, inplace=True)
                print(f"✓ {col}: Filled with median ({median_val:.2f})")

        # Strategy for categorical columns - fill with mode
        categorical_cols = self.df_clean.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if self.df_clean[col].isnull().sum() > 0:
                mode_val = self.df_clean[col].mode()[0]
                self.df_clean[col].fillna(mode_val, inplace=True)
                print(f"✓ {col}: Filled with mode ({mode_val})")

        print(f"\n✓ All missing values handled")
        return self.df_clean

    def remove_duplicates(self):
        """Remove duplicate records"""
        print("\n" + "=" * 60)
        print("STEP 4: REMOVING DUPLICATES")
        print("=" * 60)

        if self.df_clean is None:
            raise ValueError("No cleaned dataframe to deduplicate. Run handle_missing_values() first.")

        initial_count = len(self.df_clean)
        self.df_clean = self.df_clean.drop_duplicates()
        removed_count = initial_count - len(self.df_clean)

        print(f"✓ Removed {removed_count:,} duplicate records")
        print(f"✓ Remaining records: {len(self.df_clean):,}")

        return self.df_clean

    def handle_outliers(self, method="iqr"):
        """
        Handle outliers in numerical features

        Args:
            method: 'iqr' for IQR method or 'zscore' for Z-score method
        """
        print("\n" + "=" * 60)
        print("STEP 5: HANDLING OUTLIERS")
        print("=" * 60)

        if self.df_clean is None:
            raise ValueError("No cleaned dataframe available. Run previous steps first.")

        numerical_cols = self.df_clean.select_dtypes(include=["int64", "float64"]).columns
        # Exclude target variables and ID-like columns if present
        exclude_cols = ["max_monthly_emi", "requested_amount", "requested_tenure", "id"]
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]

        outlier_summary = {}

        for col in numerical_cols:
            if method == "iqr":
                Q1 = self.df_clean[col].quantile(0.25)
                Q3 = self.df_clean[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 3 * IQR  # Using 3*IQR for less aggressive addressing
                upper_bound = Q3 + 3 * IQR

                outliers = ((self.df_clean[col] < lower_bound) | (self.df_clean[col] > upper_bound)).sum()

                if outliers > 0:
                    # Cap outliers instead of removing
                    self.df_clean[col] = np.clip(self.df_clean[col], lower_bound, upper_bound)
                    outlier_summary[col] = int(outliers)

        if outlier_summary:
            print("Outliers capped (not removed to preserve data):")
            for col, count in outlier_summary.items():
                print(f"  {col}: {count:,} outliers capped")
        else:
            print("✓ No significant outliers detected")

        return self.df_clean

    def validate_data(self):
        """Validate data constraints based on domain knowledge"""
        print("\n" + "=" * 60)
        print("STEP 6: DATA VALIDATION")
        print("=" * 60)

        if self.df_clean is None:
            raise ValueError("No cleaned dataframe available. Run previous steps first.")

        validation_passed = True
        
            # Ensure required categorical columns exist for Streamlit app
        required_columns = {
        "employment_type": "Salaried",
        "company_type": "Private",
        "house_type": "Rented",
        "existing_loans": "No",
        "emi_scenario": "Standard"
    }

        for col, default in required_columns.items():
            if col not in self.df_clean.columns:
                self.df_clean[col] = default

        # Age validation
            if "age" in self.df_clean.columns:
                self.df_clean["age"] = pd.to_numeric(self.df_clean["age"], errors="coerce")
                invalid_ages = self.df_clean["age"].isna().sum()
            if invalid_ages > 0:
                print(f"⚠ Found {invalid_ages} invalid age values. Filling with median age.")
                self.df_clean["age"].fillna(self.df_clean["age"].median(), inplace=True)

            if (self.df_clean["age"] < 18).any() or (self.df_clean["age"] > 100).any():
                print("⚠ Warning: Age values outside realistic range (18-100)")
                validation_passed = False
            else:
                print("✓ Age values are valid")
        else:
            print("age column not found; skipping age validation.")

        # Credit score validation
        if "credit_score" in self.df_clean.columns:
            if (self.df_clean["credit_score"] < 300).any() or (self.df_clean["credit_score"] > 850).any():
                print("⚠ Warning: Credit score outside valid range (300-850)")
                validation_passed = False
            else:
                print("✓ Credit scores are valid")
        else:
            print("credit_score column not found; skipping credit score validation.")

        # Salary validation
        if "monthly_salary" in self.df_clean.columns:
            self.df_clean["monthly_salary"] = pd.to_numeric(
                self.df_clean["monthly_salary"], errors="coerce"
            )
            invalid_salaries = self.df_clean["monthly_salary"].isna().sum()
            if invalid_salaries > 0:
                print(f"⚠ Found {invalid_salaries} invalid monthly salary values. Filling with median salary.")
                self.df_clean["monthly_salary"].fillna(self.df_clean["monthly_salary"].median(), inplace=True)

            if (self.df_clean["monthly_salary"] <= 0).any():
                print("⚠ Warning: Negative or zero monthly salary values found")
                validation_passed = False
            else:
                print("✓ Monthly salary values are valid")
        else:
            print("monthly_salary column not found; skipping salary validation.")

        # Target variable validation
        if "emi_eligibility" in self.df_clean.columns:
            if self.df_clean["emi_eligibility"].isnull().any():
                print("⚠ Warning: Missing target values (emi_eligibility)")
                validation_passed = False
            else:
                print("✓ Target variables are complete")
        else:
            print("emi_eligibility column not found; skipping target validation.")

        if validation_passed:
            print("\n✅ All validations passed!")
        else:
            print("\n⚠ Some validations failed - review data quality")

        return validation_passed

    def create_train_test_split(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split data into train, validation, and test sets.
        Uses stratified split if possible, otherwise falls back to normal split.
        """
        print("\n" + "=" * 60)
        print("STEP 7: CREATING TRAIN-TEST-VALIDATION SPLITS")
        print("=" * 60)

        if self.df_clean is None:
            raise ValueError("No cleaned dataframe available. Run previous steps first.")

        stratify_col = None

        # Check if stratified split is possible (emi_eligibility must have at least 2 samples per class)
        if "emi_eligibility" in self.df_clean.columns:
            value_counts = self.df_clean["emi_eligibility"].value_counts()
            if value_counts.min() >= 2:
                stratify_col = self.df_clean["emi_eligibility"]
            else:
                print("⚠ Dataset too small for stratified split. Using normal split.")

        # First split: train+val vs test
        train_val, test = train_test_split(
            self.df_clean, test_size=test_size, random_state=random_state, stratify=stratify_col
        )

        # Second split: train vs validation
        val_ratio = val_size / (1 - test_size)
        stratify_train_val = None
        if stratify_col is not None:
            stratify_train_val = train_val["emi_eligibility"]

        train, val = train_test_split(
            train_val, test_size=val_ratio, random_state=random_state, stratify=stratify_train_val
        )

        print(f"✓ Training Set: {len(train):,} records ({len(train)/len(self.df_clean)*100:.1f}%)")
        print(f"✓ Validation Set: {len(val):,} records ({len(val)/len(self.df_clean)*100:.1f}%)")
        print(f"✓ Test Set: {len(test):,} records ({len(test)/len(self.df_clean)*100:.1f}%)")

        # show class distribution if available
        if "emi_eligibility" in train.columns:
            print("\nClass distribution in splits:")
            print("Training:")
            print(train["emi_eligibility"].value_counts(normalize=True))
            print("\nValidation:")
            print(val["emi_eligibility"].value_counts(normalize=True))
            print("\nTest:")
            print(test["emi_eligibility"].value_counts(normalize=True))

        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

    def save_processed_data(self, train, val, test):
        """Save processed datasets"""
        print("\n" + "=" * 60)
        print("STEP 8: SAVING PROCESSED DATA")
        print("=" * 60)

        os.makedirs("data/processed", exist_ok=True)
        train.to_csv("data/processed/train.csv", index=False)
        val.to_csv("data/processed/val.csv", index=False)
        test.to_csv("data/processed/test.csv", index=False)

        print("✓ Saved: data/processed/train.csv")
        print("✓ Saved: data/processed/val.csv")
        print("✓ Saved: data/processed/test.csv")

        print("\n" + "=" * 60)
        print("DATA PREPROCESSING COMPLETE! ✅")
        print("=" * 60)

    def run_complete_pipeline(self):
        """Execute the complete preprocessing pipeline"""
        # Step 1: Load data
        self.load_data()

        # Step 2: Assess quality
        self.assess_data_quality()

        # Step 3: Handle missing values
        self.handle_missing_values()

        # Step 4: Remove duplicates
        self.remove_duplicates()

        # Step 5: Handle outliers
        self.handle_outliers()

        # Step 6: Validate data
        self.validate_data()
        
        # ensure required analysis columns exist with sensible defaults
# put this in run_complete_pipeline(), before saving cleaned_data.csv

# helper: default requested_amount = 10 * monthly_salary (if available), else 0
        if 'requested_amount' not in self.df_clean.columns:
            if 'monthly_salary' in self.df_clean.columns:
                self.df_clean['requested_amount'] = (self.df_clean['monthly_salary'] * 10).round(0)
            else:
                self.df_clean['requested_amount'] = 0

# default tenure (months)
        if 'requested_tenure' not in self.df_clean.columns:
            self.df_clean['requested_tenure'] = 24

# Ensure numeric and no NaNs
        self.df_clean['requested_amount'] = pd.to_numeric(self.df_clean['requested_amount'], errors='coerce').fillna(
        (self.df_clean['monthly_salary'] * 10).fillna(0) if 'monthly_salary' in self.df_clean.columns else 0
            )
        self.df_clean['requested_tenure'] = pd.to_numeric(self.df_clean['requested_tenure'], errors='coerce').fillna(24).astype(int)

# other categorical defaults used by the Streamlit app (avoid KeyErrors)
        defaults = {
    'employment_type': 'Salaried',
    'company_type': 'Private',
    'emi_scenario': 'Standard',
    'emi_scenario': 'Standard'  # keep existing if present
        }
        for col, default in defaults.items():
            if col not in self.df_clean.columns:
                self.df_clean[col] = default
    # fill any NaNs
        if self.df_clean[col].isnull().any():
            self.df_clean[col].fillna(default, inplace=True)

        # Optional: Save cleaned dataset before splitting
        os.makedirs("data/processed", exist_ok=True)
        self.df_clean.to_csv("data/processed/cleaned_data.csv", index=False)
        print("✓ Saved: data/processed/cleaned_data.csv")

        # Step 7: Create splits
        train, val, test = self.create_train_test_split()

        # Step 8: Save processed data
        self.save_processed_data(train, val, test)

        return train, val, test


# Main execution
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DataPreprocessor("data/raw/emi_prediction_dataset.csv")

    # Run complete pipeline
    train, val, test = preprocessor.run_complete_pipeline()

    print("\n🎉 Ready for next section: Feature Engineering!")