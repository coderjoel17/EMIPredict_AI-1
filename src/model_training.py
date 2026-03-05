import os

os.makedirs("models/classification", exist_ok=True)
os.makedirs("models/regression", exist_ok=True)
"""
Machine Learning Model Training Module
Train and evaluate classification and regression models with MLflow tracking
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            roc_auc_score, classification_report, confusion_matrix,
                            mean_squared_error, mean_absolute_error, r2_score)
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
import warnings
warnings.filterwarnings('ignore')

class MLModelTrainer:
    """
    Complete ML training pipeline with MLflow experiment tracking
    """
    
    def __init__(self, experiment_name="EMI_Prediction"):
        """
        Initialize trainer with MLflow experiment
        
        Args:
            experiment_name: Name for MLflow experiment
        """
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        print(f"✓ MLflow experiment set: {experiment_name}")
        
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train_class = None
        self.y_val_class = None
        self.y_test_class = None
        self.y_train_reg = None
        self.y_val_reg = None
        self.y_test_reg = None
        
        self.classification_models = {}
        self.regression_models = {}
        self.best_classification_model = None
        self.best_regression_model = None
        
    def load_and_prepare_data(self):
        """Load feature-engineered data and prepare for modeling"""
        print("\n" + "="*60)
        print("STEP 1: LOADING FEATURE-ENGINEERED DATA")
        print("="*60)
        
        # Load datasets
        train = pd.read_csv('data/featured/train_features.csv')
        val = pd.read_csv('data/featured/val_features.csv')
        test = pd.read_csv('data/featured/test_features.csv')
        
        print(f"✓ Training: {len(train):,} records")
        print(f"✓ Validation: {len(val):,} records")
        print(f"✓ Test: {len(test):,} records")
        
        # --- Save target columns before encoding ---
        y_train_class_raw = train['emi_eligibility']
        y_val_class_raw = val['emi_eligibility']
        y_test_class_raw = test['emi_eligibility']
        
        y_train_reg = train['max_monthly_emi']
        y_val_reg = val['max_monthly_emi']
        y_test_reg = test['max_monthly_emi']
        
        # Drop target columns from features
        train = train.drop(['emi_eligibility', 'max_monthly_emi'], axis=1)
        val = val.drop(['emi_eligibility', 'max_monthly_emi'], axis=1)
        test = test.drop(['emi_eligibility', 'max_monthly_emi'], axis=1)
        
        categorical_cols = train.select_dtypes(include='object').columns
        print(f"Encoding {len(categorical_cols)} categorical columns: {list(categorical_cols)}")
    
        train = pd.get_dummies(train, columns=categorical_cols, drop_first=True)
        val = pd.get_dummies(val, columns=categorical_cols, drop_first=True)
        test = pd.get_dummies(test, columns=categorical_cols, drop_first=True)
    
        # Align columns (to handle mismatch between train/val/test)
        train, val = train.align(val, join='left', axis=1, fill_value=0)
        train, test = train.align(test, join='left', axis=1, fill_value=0)
        
        # --- ✅ Handle missing values ---
        train = train.fillna(0)
        val = val.fillna(0)
        test = test.fillna(0)
        
        # Prepare features
        self.X_train, self.X_val, self.X_test = train, val, test
        print(f"✓ Total features: {train.shape[1]}")

        # ---  Assign targets properly ---
        target_map = {'Eligible': 2, 'High_Risk': 1, 'Not_Eligible': 0}

        self.y_train_class = y_train_class_raw.map(target_map)
        self.y_val_class = y_val_class_raw.map(target_map)
        self.y_test_class = y_test_class_raw.map(target_map)
        
        # Prepare regression targets
        self.y_train_reg = y_train_reg
        self.y_val_reg = y_val_reg
        self.y_test_reg = y_test_reg
        
        print("\n Data preparation complete!")
        print(f"Classification target distribution:")
        print(self.y_train_class.value_counts().sort_index())
        print(f"\nRegression target statistics:")
        print(f"Mean: ₹{self.y_train_reg.mean():,.2f}")
        print(f"Range: ₹{self.y_train_reg.min():,.2f} - ₹{self.y_train_reg.max():,.2f}")
        
        # 🧩 Save feature column names for Streamlit prediction alignment
        import os
        import joblib
    
        os.makedirs("artifacts", exist_ok=True)
        
        joblib.dump(self.X_train.columns.tolist(), 'artifacts/train_columns.pkl')
        print("🗂️ Saved training feature columns to artifacts/train_columns.pkl")
        
        print("\n Data preparation complete!")
        print(f"Classification target distribution:")
        print(self.y_train_class.value_counts().sort_index())
        
    def train_classification_models(self):
        """Train multiple classification models with MLflow tracking"""
        print("\n" + "="*60)
        print("STEP 2: TRAINING CLASSIFICATION MODELS")
        print("="*60)
        
        # Define models
        models = {
            'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random_Forest': RandomForestClassifier(n_estimators=100, max_depth=15, 
                                                    random_state=42, n_jobs=-1),
            'XGBoost': XGBClassifier(n_estimators=100, max_depth=7, learning_rate=0.1,
                                    random_state=42, eval_metric='mlogloss'),
            'Decision_Tree': DecisionTreeClassifier(max_depth=15, random_state=42),
            'Gradient_Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=7,
                                                           learning_rate=0.1, random_state=42)
        }
        
        results = []
        
        for model_name, model in models.items():
            print(f"\n--- Training {model_name} ---")
            
            with mlflow.start_run(run_name=f"Classification_{model_name}"):
                # Train model
                model.fit(self.X_train, self.y_train_class)
                
                # Predictions
                y_pred_train = model.predict(self.X_train)
                y_pred_val = model.predict(self.X_val)
                
                # Metrics
                train_acc = accuracy_score(self.y_train_class, y_pred_train)
                val_acc = accuracy_score(self.y_val_class, y_pred_val)
                val_precision = precision_score(self.y_val_class, y_pred_val, average='weighted')
                val_recall = recall_score(self.y_val_class, y_pred_val, average='weighted')
                val_f1 = f1_score(self.y_val_class, y_pred_val, average='weighted')
                
                # For ROC-AUC, we need probability predictions
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(self.X_val)
                    val_roc_auc = roc_auc_score(self.y_val_class, y_pred_proba, 
                                                multi_class='ovr', average='weighted')
                else:
                    val_roc_auc = 0.0
                
                # Log parameters
                if hasattr(model, 'get_params'):
                    mlflow.log_params(model.get_params())
                
                # Log metrics
                mlflow.log_metric("train_accuracy", train_acc)
                mlflow.log_metric("val_accuracy", val_acc)
                mlflow.log_metric("val_precision", val_precision)
                mlflow.log_metric("val_recall", val_recall)
                mlflow.log_metric("val_f1_score", val_f1)
                mlflow.log_metric("val_roc_auc", val_roc_auc)
                
                # Log model
                if 'XGBoost' in model_name:
                    mlflow.xgboost.log_model(model, "model")
                else:
                    mlflow.sklearn.log_model(model, "model")
                
                # Store model
                self.classification_models[model_name] = model
                
                # Print results
                print(f"✓ Training Accuracy: {train_acc:.4f}")
                print(f"✓ Validation Accuracy: {val_acc:.4f}")
                print(f"✓ Precision: {val_precision:.4f}")
                print(f"✓ Recall: {val_recall:.4f}")
                print(f"✓ F1-Score: {val_f1:.4f}")
                print(f"✓ ROC-AUC: {val_roc_auc:.4f}")
                
                results.append({
                    'Model': model_name,
                    'Train_Acc': train_acc,
                    'Val_Acc': val_acc,
                    'Precision': val_precision,
                    'Recall': val_recall,
                    'F1_Score': val_f1,
                    'ROC_AUC': val_roc_auc
                })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results).sort_values('Val_Acc', ascending=False)
        print("\n" + "="*60)
        print("CLASSIFICATION MODELS COMPARISON")
        print("="*60)
        print(results_df.to_string(index=False))
        
        # Save comparison
        results_df.to_csv('reports/classification_comparison.csv', index=False)
        print("\n✓ Saved: reports/classification_comparison.csv")
        
        # Select best model
        best_model_name = results_df.iloc[0]['Model']
        self.best_classification_model = self.classification_models[best_model_name]
        print(f"\n🏆 Best Classification Model: {best_model_name}")
        print(f"   Validation Accuracy: {results_df.iloc[0]['Val_Acc']:.4f}")
        
        return results_df
    
    def train_regression_models(self):
        """Train multiple regression models with MLflow tracking"""
        print("\n" + "="*60)
        print("STEP 3: TRAINING REGRESSION MODELS")
        print("="*60)
        
        # Define models
        models = {
            'Linear_Regression': LinearRegression(),
            'Random_Forest': RandomForestRegressor(n_estimators=100, max_depth=15,
                                                   random_state=42, n_jobs=-1),
            'XGBoost': XGBRegressor(n_estimators=100, max_depth=7, learning_rate=0.1,
                                   random_state=42),
            'Decision_Tree': DecisionTreeRegressor(max_depth=15, random_state=42),
            'Gradient_Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=7,
                                                           learning_rate=0.1, random_state=42)
        }
        
        results = []
        
        for model_name, model in models.items():
            print(f"\n--- Training {model_name} ---")
            
            with mlflow.start_run(run_name=f"Regression_{model_name}"):
                # Train model
                model.fit(self.X_train, self.y_train_reg)
                
                # Predictions
                y_pred_train = model.predict(self.X_train)
                y_pred_val = model.predict(self.X_val)
                
                # Metrics
                train_rmse = np.sqrt(mean_squared_error(self.y_train_reg, y_pred_train))
                val_rmse = np.sqrt(mean_squared_error(self.y_val_reg, y_pred_val))
                val_mae = mean_absolute_error(self.y_val_reg, y_pred_val)
                val_r2 = r2_score(self.y_val_reg, y_pred_val)
                
                # MAPE (Mean Absolute Percentage Error)
                val_mape = np.mean(np.abs((self.y_val_reg - y_pred_val) / self.y_val_reg)) * 100
                
                # Log parameters
                if hasattr(model, 'get_params'):
                    mlflow.log_params(model.get_params())
                
                # Log metrics
                mlflow.log_metric("train_rmse", train_rmse)
                mlflow.log_metric("val_rmse", val_rmse)
                mlflow.log_metric("val_mae", val_mae)
                mlflow.log_metric("val_r2", val_r2)
                mlflow.log_metric("val_mape", val_mape)
                
                # Log model
                if 'XGBoost' in model_name:
                    mlflow.xgboost.log_model(model, "model")
                else:
                    mlflow.sklearn.log_model(model, "model")
                
                # Store model
                self.regression_models[model_name] = model
                
                # Print results
                print(f"✓ Training RMSE: ₹{train_rmse:,.2f}")
                print(f"✓ Validation RMSE: ₹{val_rmse:,.2f}")
                print(f"✓ MAE: ₹{val_mae:,.2f}")
                print(f"✓ R² Score: {val_r2:.4f}")
                print(f"✓ MAPE: {val_mape:.2f}%")
                
                results.append({
                    'Model': model_name,
                    'Train_RMSE': train_rmse,
                    'Val_RMSE': val_rmse,
                    'MAE': val_mae,
                    'R2_Score': val_r2,
                    'MAPE': val_mape
                })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results).sort_values('Val_RMSE', ascending=True)
        print("\n" + "="*60)
        print("REGRESSION MODELS COMPARISON")
        print("="*60)
        print(results_df.to_string(index=False))
        
        # Save comparison
        results_df.to_csv('reports/regression_comparison.csv', index=False)
        print("\n✓ Saved: reports/regression_comparison.csv")
        
        # Select best model
        best_model_name = results_df.iloc[0]['Model']
        self.best_regression_model = self.regression_models[best_model_name]
        print(f"\n🏆 Best Regression Model: {best_model_name}")
        print(f"   Validation RMSE: ₹{results_df.iloc[0]['Val_RMSE']:,.2f}")
        print(f"   R² Score: {results_df.iloc[0]['R2_Score']:.4f}")
        
        return results_df
    
    def evaluate_best_models_on_test(self):
        """Evaluate best models on test set"""
        print("\n" + "="*60)
        print("STEP 4: FINAL EVALUATION ON TEST SET")
        print("="*60)
        
        # Classification
        print("\n--- Classification Model ---")
        y_pred_class = self.best_classification_model.predict(self.X_test)
        
        test_acc = accuracy_score(self.y_test_class, y_pred_class)
        test_precision = precision_score(self.y_test_class, y_pred_class, average='weighted')
        test_recall = recall_score(self.y_test_class, y_pred_class, average='weighted')
        test_f1 = f1_score(self.y_test_class, y_pred_class, average='weighted')
        
        print(f"✓ Test Accuracy: {test_acc:.4f}")
        print(f"✓ Precision: {test_precision:.4f}")
        print(f"✓ Recall: {test_recall:.4f}")
        print(f"✓ F1-Score: {test_f1:.4f}")
        
        print("\nClassification Report:")
        target_names = ['Not_Eligible', 'High_Risk', 'Eligible']
        print(classification_report(self.y_test_class, y_pred_class, target_names=target_names))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(self.y_test_class, y_pred_class)
        print(cm)
        
        # Regression
        print("\n--- Regression Model ---")
        y_pred_reg = self.best_regression_model.predict(self.X_test)
        
        test_rmse = np.sqrt(mean_squared_error(self.y_test_reg, y_pred_reg))
        test_mae = mean_absolute_error(self.y_test_reg, y_pred_reg)
        test_r2 = r2_score(self.y_test_reg, y_pred_reg)
        test_mape = np.mean(np.abs((self.y_test_reg - y_pred_reg) / self.y_test_reg)) * 100
        
        print(f"✓ Test RMSE: ₹{test_rmse:,.2f}")
        print(f"✓ MAE: ₹{test_mae:,.2f}")
        print(f"✓ R² Score: {test_r2:.4f}")
        print(f"✓ MAPE: {test_mape:.2f}%")
        
        # Save test results
        test_results = {
            'Classification': {
                'Accuracy': test_acc,
                'Precision': test_precision,
                'Recall': test_recall,
                'F1_Score': test_f1
            },
            'Regression': {
                'RMSE': test_rmse,
                'MAE': test_mae,
                'R2_Score': test_r2,
                'MAPE': test_mape
            }
        }
        
        pd.DataFrame(test_results).to_csv('reports/test_results.csv')
        print("\n✓ Saved: reports/test_results.csv")
        
    def save_best_models(self):
        """Save best models for deployment"""
        print("\n" + "="*60)
        print("STEP 5: SAVING BEST MODELS")
        print("="*60)

        import os

        # Ensure directories exist
        os.makedirs("models/classification", exist_ok=True)
        os.makedirs("models/regression", exist_ok=True)

        # Save classification model
        joblib.dump(
            self.best_classification_model,
            "models/classification/best_model.pkl"
            )
        print("✓ Saved: models/classification/best_model.pkl")

        # Save regression model
        joblib.dump(
            self.best_regression_model,
            "models/regression/best_model.pkl"
            )
        print("✓ Saved: models/regression/best_model.pkl")

        # Save feature names
        feature_names = self.X_train.columns.tolist()
        pd.DataFrame({'features': feature_names}).to_csv(
            "models/feature_names.csv", index=False
            )
        print("✓ Saved: models/feature_names.csv")
        
    def run_complete_training_pipeline(self):
        """Execute complete model training pipeline"""
        print("\n" + "🤖" + "="*58 + "🤖")
        print("  STARTING MACHINE LEARNING TRAINING PIPELINE")
        print("🤖" + "="*58 + "🤖" + "\n")
        
        # Step 1: Load data
        self.load_and_prepare_data()
        
        # Step 2: Train classification models
        class_results = self.train_classification_models()
        
        # Step 3: Train regression models
        reg_results = self.train_regression_models()
        
        # Step 4: Evaluate on test set
        self.evaluate_best_models_on_test()
        
        # Step 5: Save models
        self.save_best_models()
        
        print("\n" + "="*60)
        print("✅ MODEL TRAINING COMPLETE!")
        print("="*60)
        print("\n📊 View MLflow dashboard with: mlflow ui")
        print("📊 Open browser at: http://localhost:5000")
        print("\n🎉 Ready for next section: Streamlit Application!")


# Main execution
if __name__ == "__main__":
    # Initialize trainer
    trainer = MLModelTrainer(experiment_name="EMI_Prediction")
    
    # Run complete pipeline
    trainer.run_complete_training_pipeline()