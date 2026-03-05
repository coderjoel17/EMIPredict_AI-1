"""
Feature Engineering Module
Create advanced features for EMI prediction models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
import warnings

warnings.filterwarnings("ignore")


class FeatureEngineer:
    """
    Advanced feature engineering for financial risk assessment
    """

    def __init__(self):
        """Initialize feature engineer"""
        self.label_encoders = {}
        self.scaler = None
        self.feature_names = []

    def load_data(self, train_path, val_path, test_path):
        """Load preprocessed datasets"""
        print("=" * 60)
        print("LOADING PROCESSED DATA")
        print("=" * 60)

        self.train = pd.read_csv(train_path)
        self.val = pd.read_csv(val_path)
        self.test = pd.read_csv(test_path)

        print(f"✓ Training: {len(self.train):,} records")
        print(f"✓ Validation: {len(self.val):,} records")
        print(f"✓ Test: {len(self.test):,} records")

        return self.train, self.val, self.test
    
    def create_financial_ratios(self, df):
        """
        Create derived financial ratios.
        Handles missing columns by creating sensible defaults so the pipeline
        can run on small/demo datasets without crashing.
        """
        print("\n" + "=" * 60)
        print("1. CREATING FINANCIAL RATIOS")
        print("=" * 60)

    # Ensure monthly_salary exists and is numeric (fallback to 0)
        df["monthly_salary"] = pd.to_numeric(df.get("monthly_salary", 0), errors="coerce").fillna(0)

    # Defaults for other numeric columns (use monthly_salary-based heuristics where useful)
        if "current_emi_amount" not in df.columns:
            # assume current EMI is ~20% of salary for demo rows
            df["current_emi_amount"] = (df["monthly_salary"] * 0.2).fillna(0)

        if "requested_amount" not in df.columns:
        # assume requested amount equals ~10 months of salary as a reasonable default
            df["requested_amount"] = (df["monthly_salary"] * 10).fillna(0)

        if "requested_tenure" not in df.columns:
            df["requested_tenure"] = 24  # months

    # Expense-related defaults (missing expense columns -> 0)
        expense_cols = [
            "monthly_rent",
            "school_fees",
            "college_fees",
            "travel_expenses",
            "groceries_utilities",
            "other_monthly_expenses",
        ]
        for col in expense_cols:
            if col not in df.columns:
                df[col] = 0
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Banking/savings defaults
        if "bank_balance" not in df.columns:
            df["bank_balance"] = 0
        else:
            df["bank_balance"] = pd.to_numeric(df["bank_balance"], errors="coerce").fillna(0)

        if "emergency_fund" not in df.columns:
            df["emergency_fund"] = 0
        else:
            df["emergency_fund"] = pd.to_numeric(df["emergency_fund"], errors="coerce").fillna(0)

    # Ensure requested_amount/requested_tenure numeric
        df["requested_amount"] = pd.to_numeric(df["requested_amount"], errors="coerce").fillna(0)
        df["requested_tenure"] = pd.to_numeric(df["requested_tenure"], errors="coerce").fillna(0)

    # --- Create features safely (avoid division by zero) ---
    # 1. Debt-to-Income Ratio (DTI)
        df["debt_to_income_ratio"] = np.where(
            df["monthly_salary"] > 0,
            (df["current_emi_amount"] / df["monthly_salary"] * 100),
            0,
            )
        df["debt_to_income_ratio"] = df["debt_to_income_ratio"].clip(0, 100)
        print("✓ Created: debt_to_income_ratio")

    # 2. Total Expense Ratio & total_monthly_expenses
        df["total_monthly_expenses"] = df[expense_cols].sum(axis=1)
        df["expense_to_income_ratio"] = np.where(
            df["monthly_salary"] > 0,
            (df["total_monthly_expenses"] / df["monthly_salary"] * 100),
            0,
        )
        df["expense_to_income_ratio"] = df["expense_to_income_ratio"].clip(0, 150)
        print("✓ Created: total_monthly_expenses, expense_to_income_ratio")

    # 3. Disposable Income (salary - expenses - current EMI)
        df["disposable_income"] = df["monthly_salary"] - df["total_monthly_expenses"] - df["current_emi_amount"]
        df["disposable_income"] = df["disposable_income"].fillna(0).clip(lower=0)
        print("✓ Created: disposable_income")

    # 4. Affordability Ratio (percentage of income available for new EMI)
        df["affordability_ratio"] = np.where(
            df["monthly_salary"] > 0,
            (df["disposable_income"] / df["monthly_salary"] * 100),
            0,
        )
        df["affordability_ratio"] = df["affordability_ratio"].clip(0, 100)
        print("✓ Created: affordability_ratio")

    # 5. Total Obligation Ratio
        df["total_obligations"] = df["current_emi_amount"] + df["total_monthly_expenses"]
        df["obligation_ratio"] = np.where(
            df["monthly_salary"] > 0,
            (df["total_obligations"] / df["monthly_salary"] * 100),
            0,
        )
        df["obligation_ratio"] = df["obligation_ratio"].clip(0, 150)
        print("✓ Created: obligation_ratio")

    # 6. Savings Ratio (savings relative to monthly salary)
        df["savings_ratio"] = np.where(
            df["monthly_salary"] > 0,
            ((df["bank_balance"] + df["emergency_fund"]) / df["monthly_salary"]) * 100,
            0,
        )
        df["savings_ratio"] = df["savings_ratio"].clip(0, 5000)  # generous upper bound for safety
        print("✓ Created: savings_ratio")

    # 7. EMI-to-Requested Amount Ratio (expected new EMI burden)
    # compute monthly_new_emi safely
        monthly_new_emi = np.where(
            df["requested_tenure"] > 0,
            df["requested_amount"] / df["requested_tenure"],
            0,
        )
        df["emi_burden_ratio"] = np.where(
            df["monthly_salary"] > 0,
            (monthly_new_emi / df["monthly_salary"] * 100),
            0,
        )
        df["emi_burden_ratio"] = df["emi_burden_ratio"].clip(0, 100)
        print("✓ Created: emi_burden_ratio")

        print(f"\n✅ Created financial ratio features (available columns: {len(df.columns)})")
        return df

    def create_risk_scoring_features(self, df):

        print("\n" + "=" * 60)
        print("2. CREATING RISK SCORING FEATURES")
        print("=" * 60)

    # Ensure required columns exist
        if "years_of_employment" not in df.columns:
            df["years_of_employment"] = 1

        if "dependents" not in df.columns:
            df["dependents"] = 0

        if "family_size" not in df.columns:
            df["family_size"] = 1

        if "house_type" not in df.columns:
            df["house_type"] = "Rented"

        if "existing_loans" not in df.columns:
            df["existing_loans"] = "No"

    # 1. Credit Risk Score
        df["credit_risk_score"] = (df["credit_score"] - 300) / (850 - 300) * 100
        print("✓ Created: credit_risk_score")

    # 2. Employment Stability Score
        df["employment_stability_score"] = np.minimum(
        df["years_of_employment"] * 10, 100
            )
        print("✓ Created: employment_stability_score")

    # 3. Financial Cushion Score
        df["financial_cushion_score"] = (
            (df["savings_ratio"] * 0.6) +
            ((100 - df["obligation_ratio"]) * 0.4)
            ).clip(0, 100)
        print("✓ Created: financial_cushion_score")

    # 4. Family Burden Score
        df["family_burden_score"] = (
            df["dependents"] / df["family_size"] * 100
            ).fillna(0).clip(0, 100)
        print("✓ Created: family_burden_score")

    # 5. Housing Stability
        housing_map = {"Own": 100, "Family": 80, "Rented": 50}
        df["housing_stability_score"] = df["house_type"].map(housing_map).fillna(50)
        print("✓ Created: housing_stability_score")

    # 6. Existing Loans Indicator
        df["has_existing_loans"] = (df["existing_loans"] == "Yes").astype(int)
        print("✓ Created: has_existing_loans")

        print("\n✅ Created 6 risk scoring features")

        return df
    def create_interaction_features(self, df):
        """
        Create interaction features between important variables
        Captures relationships between features
        """
        print("\n" + "=" * 60)
        print("3. CREATING INTERACTION FEATURES")
        print("=" * 60)

        # 1. Income × Credit Score interaction
        df["income_credit_interaction"] = (df["monthly_salary"] / 100000) * (
            df["credit_score"] / 100
        )
        print("✓ Created: income_credit_interaction")

        # 2. Age × Employment interaction
        df["age_employment_interaction"] = df["age"] * df["years_of_employment"]
        print("✓ Created: age_employment_interaction")

        # 3. Savings × Credit interaction
        df["savings_credit_interaction"] = (df["bank_balance"] / 10000) * (
            df["credit_score"] / 100
        )
        print("✓ Created: savings_credit_interaction")

        # 4. Income per family member
        df["per_capita_income"] = df["monthly_salary"] / df["family_size"]
        print("✓ Created: per_capita_income")

        # 5. Emergency fund sufficiency (months of expenses covered)
        df["emergency_fund_months"] = (
            df["emergency_fund"] / (df["total_monthly_expenses"] + 1)
        ).clip(0, 24)
        print("✓ Created: emergency_fund_months")

        print(f"\n✅ Created 5 interaction features")
        return df

    def create_categorical_features(self, df):
        """
        Create additional categorical features
        """
        print("\n" + "=" * 60)
        print("4. CREATING CATEGORICAL FEATURES")
        print("=" * 60)

        # 1. Age groups
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 30, 40, 50, 100],
            labels=["Young", "Mid", "Senior", "Elder"],
        )
        print("✓ Created: age_group")

        # 2. Income brackets
        df["income_bracket"] = pd.cut(
            df["monthly_salary"],
            bins=[0, 30000, 50000, 75000, 100000, float("inf")],
            labels=["Low", "Medium", "Upper-Mid", "High", "Very-High"],
        )
        print("✓ Created: income_bracket")

        # 3. Credit score categories
        df["credit_category"] = pd.cut(
            df["credit_score"],
            bins=[0, 580, 670, 740, 800, 850],
            labels=["Poor", "Fair", "Good", "Very-Good", "Excellent"],
        )
        print("✓ Created: credit_category")

        # 4. Risk level based on obligation ratio
        df["risk_level"] = pd.cut(
            df["obligation_ratio"],
            bins=[0, 40, 60, 80, 150],
            labels=["Low-Risk", "Medium-Risk", "High-Risk", "Very-High-Risk"],
        )
        print("✓ Created: risk_level")

        print(f"\n✅ Created 4 categorical features")
        return df

    def encode_categorical_variables(self, train_df, val_df, test_df):
        """
        Encode categorical variables using Label Encoding
        """
        print("\n" + "=" * 60)
        print("5. ENCODING CATEGORICAL VARIABLES")
        print("=" * 60)

        categorical_cols = [
            "gender",
            "marital_status",
            "education",
            "employment_type",
            "company_type",
            "house_type",
            "existing_loans",
            "emi_scenario",
            "age_group",
            "income_bracket",
            "credit_category",
            "risk_level",
        ]

        for col in categorical_cols:
            if col in train_df.columns:

                le = LabelEncoder()

                # combine all datasets to learn every category
                combined = pd.concat([
                train_df[col],
                val_df[col],
                test_df[col]
                ]).astype(str)

                le.fit(combined)

                train_df[f"{col}_encoded"] = le.transform(train_df[col].astype(str))
                val_df[f"{col}_encoded"] = le.transform(val_df[col].astype(str))
                test_df[f"{col}_encoded"] = le.transform(test_df[col].astype(str))

                self.label_encoders[col] = le

                print(f"✓ Encoded: {col}")

                print(f"\n✅ Encoded {len(categorical_cols)} categorical features")
                return train_df, val_df, test_df

    def scale_numerical_features(self, train_df, val_df, test_df):
        """
        Scale numerical features using RobustScaler
        RobustScaler is better for data with outliers
        """
        print("\n" + "=" * 60)
        print("6. SCALING NUMERICAL FEATURES")
        print("=" * 60)

        # Select numerical columns (exclude targets and encoded columns)
        numerical_cols = train_df.select_dtypes(include=["int64", "float64"]).columns
        exclude_cols = ["emi_eligibility", "max_monthly_emi"]
        numerical_cols = [
            col
            for col in numerical_cols
            if col not in exclude_cols and not col.endswith("_encoded")
        ]

        # Initialize scaler
        self.scaler = RobustScaler()

        # Fit on training data and transform all sets
        train_df[numerical_cols] = self.scaler.fit_transform(train_df[numerical_cols])
        val_df[numerical_cols] = self.scaler.transform(val_df[numerical_cols])
        test_df[numerical_cols] = self.scaler.transform(test_df[numerical_cols])

        print(f"✓ Scaled {len(numerical_cols)} numerical features")
        print(f"✓ Using RobustScaler (robust to outliers)")

        return train_df, val_df, test_df

    def prepare_model_inputs(self, df, target_type="classification"):
        """
        Prepare final feature matrix and target variable

        Args:
            df: DataFrame with all features
            target_type: 'classification' or 'regression'
        """
        # Drop original categorical columns (keep encoded versions)
        cols_to_drop = [
            "gender",
            "marital_status",
            "education",
            "employment_type",
            "company_type",
            "house_type",
            "existing_loans",
            "emi_scenario",
            "age_group",
            "income_bracket",
            "credit_category",
            "risk_level",
        ]

        # Also drop target columns
        cols_to_drop.extend(["emi_eligibility", "max_monthly_emi"])

        # Keep only columns that exist
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]

        # Create feature matrix
        X = df.drop(columns=cols_to_drop)

        # Create target variable
        if target_type == "classification":
            # Encode target classes: Eligible=2, High_Risk=1, Not_Eligible=0
            target_map = {"Eligible": 2, "High_Risk": 1, "Not_Eligible": 0}
            y = df["emi_eligibility"].map(target_map)
        else:  # regression
            y = df["max_monthly_emi"]

        return X, y

    def run_complete_feature_engineering(self, train_path, val_path, test_path):
        """Execute complete feature engineering pipeline"""
        print("\n" + "🔧" + "=" * 58 + "🔧")
        print("  STARTING FEATURE ENGINEERING PIPELINE")
        print("🔧" + "=" * 58 + "🔧" + "\n")

        # Load data
        train, val, test = self.load_data(train_path, val_path, test_path)

        # Apply feature engineering to all datasets
        print("\n📊 Applying transformations to all datasets...")

        for df_name, df in [("Training", train), ("Validation", val), ("Test", test)]:
            print(f"\n--- Processing {df_name} Set ---")
            df = self.create_financial_ratios(df)
            df = self.create_risk_scoring_features(df)
            df = self.create_interaction_features(df)
            df = self.create_categorical_features(df)

        # Encode categorical variables
        train, val, test = self.encode_categorical_variables(train, val, test)

        # Scale numerical features
        train, val, test = self.scale_numerical_features(train, val, test)

        # Save feature-engineered datasets
        print("\n" + "=" * 60)
        print("7. SAVING FEATURE-ENGINEERED DATA")
        print("=" * 60)

        train.to_csv("data/featured/train_features.csv", index=False)
        val.to_csv("data/featured/val_features.csv", index=False)
        test.to_csv("data/featured/test_features.csv", index=False)

        print("✓ Saved: data/featured/train_features.csv")
        print("✓ Saved: data/featured/val_features.csv")
        print("✓ Saved: data/featured/test_features.csv")

        # Generate feature summary
        self.generate_feature_summary(train)

        print("\n" + "=" * 60)
        print("✅ FEATURE ENGINEERING COMPLETE!")
        print("=" * 60)
        print(
            f"\n📊 Total Features Created: {len(train.columns) - 2}"
        )  # Excluding targets
        print(f"📊 Original Features: 22")
        print(f"📊 New Features: {len(train.columns) - 24}")
        print("\n🎉 Ready for next section: Model Training!")

        return train, val, test

    def generate_feature_summary(self, df):
        """Generate summary of created features"""
        print("\n" + "=" * 60)
        print("FEATURE SUMMARY")
        print("=" * 60)

        summary = {
            "Financial Ratios": [
                "debt_to_income_ratio",
                "total_monthly_expenses",
                "expense_to_income_ratio",
                "disposable_income",
                "affordability_ratio",
                "obligation_ratio",
                "savings_ratio",
                "emi_burden_ratio",
            ],
            "Risk Scores": [
                "credit_risk_score",
                "employment_stability_score",
                "financial_cushion_score",
                "family_burden_score",
                "housing_stability_score",
                "has_existing_loans",
            ],
            "Interaction Features": [
                "income_credit_interaction",
                "age_employment_interaction",
                "savings_credit_interaction",
                "per_capita_income",
                "emergency_fund_months",
            ],
            "Categorical Features": [
                "age_group_encoded",
                "income_bracket_encoded",
                "credit_category_encoded",
                "risk_level_encoded",
            ],
        }

        report_lines = ["# Feature Engineering Summary\n"]
        report_lines.append(f"## Overview\n")
        report_lines.append(f"- **Total Features**: {len(df.columns) - 2}\n")
        report_lines.append(f"- **Dataset Size**: {len(df):,} records\n\n")

        for category, features in summary.items():
            report_lines.append(f"## {category}\n")
            existing_features = [f for f in features if f in df.columns]
            for feature in existing_features:
                if df[feature].dtype in ["int64", "float64"]:
                    stats = df[feature].describe()
                    report_lines.append(f"- **{feature}**\n")
                    report_lines.append(f"  - Mean: {stats['mean']:.2f}\n")
                    report_lines.append(f"  - Std: {stats['std']:.2f}\n")
                    report_lines.append(
                        f"  - Range: [{stats['min']:.2f}, {stats['max']:.2f}]\n"
                    )
            report_lines.append("\n")

        # Save report
        with open("reports/feature_engineering_summary.md", "w") as f:
            f.writelines(report_lines)

        print("✓ Saved: reports/feature_engineering_summary.md")


# Main execution
if __name__ == "__main__":
    # Initialize feature engineer
    engineer = FeatureEngineer()

    # Run complete pipeline
    train, val, test = engineer.run_complete_feature_engineering(
        "data/processed/train.csv", "data/processed/val.csv", "data/processed/test.csv"
    )
