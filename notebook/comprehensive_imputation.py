import pandas as pd
import numpy as np

# ✅ Must enable experimental IterativeImputer before importing it
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class ComprehensiveImputation:
    def __init__(self, data: pd.DataFrame):
        self.original_data = data.copy()
        self.data = data.copy()
        self.imputed_datasets = {}
        self.evaluation_results = {}

    def analyze_missing_data(self) -> Dict[str, Any]:
        missing_info = {
            'total_rows': len(self.data),
            'missing_by_column': self.data.isnull().sum(),
            'missing_percentage': (self.data.isnull().sum() / len(self.data)) * 100,
            'rows_with_missing': self.data.isnull().any(axis=1).sum(),
            'complete_rows': len(self.data) - self.data.isnull().any(axis=1).sum()
        }
        print("=== MISSING DATA ANALYSIS ===")
        print(f"Total rows: {missing_info['total_rows']}")
        print(f"Complete rows: {missing_info['complete_rows']}")
        print(f"Rows with missing data: {missing_info['rows_with_missing']}")
        print("\nMissing data by column:")
        for col, count in missing_info['missing_by_column'].items():
            if count > 0:
                pct = missing_info['missing_percentage'][col]
                print(f"  {col}: {count} ({pct:.1f}%)")
        return missing_info

    def simple_imputation(self) -> pd.DataFrame:
        data_simple = self.data.copy()
        numerical_cols = data_simple.select_dtypes(include=[np.number]).columns
        categorical_cols = data_simple.select_dtypes(include=['object', 'category']).columns

        for col in numerical_cols:
            if data_simple[col].isnull().any():
                mean_value = data_simple[col].mean()
                data_simple[col].fillna(mean_value, inplace=True)
                print(f"Filled {col} with mean: {mean_value:.2f}")

        for col in categorical_cols:
            if data_simple[col].isnull().any():
                mode_value = data_simple[col].mode()
                if len(mode_value) > 0:
                    data_simple[col].fillna(mode_value[0], inplace=True)
                    print(f"Filled {col} with mode: {mode_value[0]}")
                else:
                    data_simple[col].fillna('Unknown', inplace=True)
                    print(f"Filled {col} with 'Unknown' (no mode found)")

        self.imputed_datasets['simple'] = data_simple
        return data_simple

    def statistical_imputation(self) -> pd.DataFrame:
        data_stat = self.data.copy()
        numerical_cols = data_stat.select_dtypes(include=[np.number]).columns
        categorical_cols = data_stat.select_dtypes(include=['object', 'category']).columns

        for col in numerical_cols:
            if data_stat[col].isnull().any():
                skewness = data_stat[col].skew()
                if abs(skewness) > 1:
                    fill_value = data_stat[col].median()
                    method = "median"
                else:
                    fill_value = data_stat[col].mean()
                    method = "mean"
                data_stat[col].fillna(fill_value, inplace=True)
                print(f"Filled {col} with {method}: {fill_value:.2f} (skew: {skewness:.2f})")

        for col in categorical_cols:
            if data_stat[col].isnull().any():
                data_stat[col].fillna('Missing', inplace=True)
                print(f"Filled {col} with 'Missing' category")

        self.imputed_datasets['statistical'] = data_stat
        return data_stat

    def knn_imputation(self, n_neighbors: int = 3) -> pd.DataFrame:
        data_knn = self.data.copy()

        # Encode categoricals
        label_encoders = {}
        categorical_cols = data_knn.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            temp_data = data_knn[col].fillna('TEMP_MISSING')
            data_knn[col + '_encoded'] = le.fit_transform(temp_data)
            label_encoders[col] = le
            missing_mask = data_knn[col].isnull()
            data_knn.loc[missing_mask, col + '_encoded'] = np.nan

        # Numeric features for KNN (includes encoded cols automatically, no manual concat)
        numerical_cols = data_knn.select_dtypes(include=[np.number]).columns
        knn_data = data_knn[numerical_cols].copy()

        imputer = KNNImputer(n_neighbors=n_neighbors)
        knn_imputed = pd.DataFrame(
            imputer.fit_transform(knn_data),
            columns=knn_data.columns,
            index=knn_data.index
        )

        data_knn[numerical_cols] = knn_imputed[numerical_cols]

        # Decode back
        for col in categorical_cols:
            encoded_col = col + '_encoded'
            if encoded_col in knn_imputed.columns:
                rounded = np.round(knn_imputed[encoded_col]).astype(int)
                max_cat = len(label_encoders[col].classes_) - 1
                rounded = np.clip(rounded, 0, max_cat)
                data_knn[col] = label_encoders[col].inverse_transform(rounded)
                data_knn.drop(columns=[encoded_col], inplace=True)

        print(f"Applied KNN imputation with {n_neighbors} neighbors")
        self.imputed_datasets['knn'] = data_knn
        return data_knn

    def iterative_imputation(self, max_iter: int = 10) -> pd.DataFrame:
        data_iter = self.data.copy()

        # Encode categoricals
        label_encoders = {}
        categorical_cols = data_iter.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            temp = data_iter[col].fillna('TEMP_MISSING')
            data_iter[col + '_encoded'] = le.fit_transform(temp)
            label_encoders[col] = le
            missing_mask = data_iter[col].isnull()
            data_iter.loc[missing_mask, col + '_encoded'] = np.nan

        # ✅ Use only numeric columns (already includes *_encoded). Do NOT append them again.
        numerical_cols = data_iter.select_dtypes(include=[np.number]).columns
        iter_data = data_iter[numerical_cols].copy()

        imputer = IterativeImputer(max_iter=max_iter, random_state=42)
        iter_array = imputer.fit_transform(iter_data)
        iter_imputed = pd.DataFrame(iter_array, columns=iter_data.columns, index=iter_data.index)

        # ✅ Block assignment avoids shape/key mismatches
        data_iter[numerical_cols] = iter_imputed[numerical_cols]

        # Decode back
        for col in categorical_cols:
            encoded_col = col + '_encoded'
            if encoded_col in iter_imputed.columns:
                rounded = np.round(iter_imputed[encoded_col]).astype(int)
                max_cat = len(label_encoders[col].classes_) - 1
                rounded = np.clip(rounded, 0, max_cat)
                data_iter[col] = label_encoders[col].inverse_transform(rounded)
                data_iter.drop(columns=[encoded_col], inplace=True)

        print(f"Applied iterative imputation with {max_iter} iterations")
        self.imputed_datasets['iterative'] = data_iter
        return data_iter

    def advanced_imputation(self) -> pd.DataFrame:
        data_advanced = self.data.copy()

        for col in data_advanced.columns:
            if not data_advanced[col].isnull().any():
                continue

            print(f"Imputing column: {col}")
            missing_mask = data_advanced[col].isnull()
            train_data = data_advanced[~missing_mask].copy()
            predict_data = data_advanced[missing_mask].copy()

            if len(train_data) == 0:
                print(f"  No data to train on for {col}, using simple imputation")
                if data_advanced[col].dtype in ['object', 'category']:
                    data_advanced[col].fillna('Unknown', inplace=True)
                else:
                    data_advanced[col].fillna(data_advanced[col].mean(), inplace=True)
                continue

            feature_cols = [c for c in data_advanced.columns if c != col]
            X_train = train_data[feature_cols].copy()
            X_predict = predict_data[feature_cols].copy()

            # Encode categorical features
            for feat_col in feature_cols:
                if X_train[feat_col].dtype in ['object', 'category']:
                    combined = pd.concat([X_train[feat_col], X_predict[feat_col]])
                    le_feat = LabelEncoder()
                    le_feat.fit(combined.fillna('Missing'))
                    X_train[feat_col] = le_feat.transform(X_train[feat_col].fillna('Missing'))
                    X_predict[feat_col] = le_feat.transform(X_predict[feat_col].fillna('Missing'))

            # Fill any remaining NaNs in features using training means
            X_train = X_train.fillna(X_train.mean(numeric_only=True))
            X_predict = X_predict.fillna(X_train.mean(numeric_only=True))

            y_train = train_data[col]

            if data_advanced[col].dtype in ['object', 'category']:
                model = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42)

            model.fit(X_train, y_train)
            preds = model.predict(X_predict)
            data_advanced.loc[missing_mask, col] = preds
            print(f"  Filled {missing_mask.sum()} missing values")

        self.imputed_datasets['advanced'] = data_advanced
        return data_advanced

    def compare_methods(self) -> pd.DataFrame:
        if not self.imputed_datasets:
            print("No imputation methods have been applied yet.")
            return None

        rows = []
        for method, data in self.imputed_datasets.items():
            d = {
                'Method': method,
                'Remaining_Missing': data.isnull().sum().sum(),
                'Complete_Rows': len(data) - data.isnull().any(axis=1).sum()
            }
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                d['Avg_Numerical_Mean'] = data[numerical_cols].mean().mean()
                d['Avg_Numerical_Std'] = data[numerical_cols].std().mean()
            rows.append(d)

        df = pd.DataFrame(rows)
        print("\n=== IMPUTATION METHODS COMPARISON ===")
        print(df.to_string(index=False))
        return df

    def visualize_imputation_results(self):
        if not self.imputed_datasets:
            print("No imputation methods have been applied yet.")
            return

        n_methods = len(self.imputed_datasets)
        fig, axes = plt.subplots(2, n_methods, figsize=(4*n_methods, 10))
        if n_methods == 1:
            axes = axes.reshape(2, 1)

        numerical_cols = self.original_data.select_dtypes(include=[np.number]).columns
        target_col = None
        for col in numerical_cols:
            if self.original_data[col].isnull().any():
                target_col = col
                break
        if target_col is None:
            print("No numerical column with missing values found for visualization.")
            return

        for i, (method, data) in enumerate(self.imputed_datasets.items()):
            axes[0, i].hist(self.original_data[target_col].dropna(), alpha=0.5, label='Original', bins=20, density=True)
            axes[0, i].hist(data[target_col], alpha=0.7, label='Imputed', bins=20, density=True)
            axes[0, i].set_title(f'{method.title()} - {target_col} Distribution')
            axes[0, i].legend()

            sns.heatmap(data.isnull(), cbar=True, ax=axes[1, i], cmap='viridis')
            axes[1, i].set_title(f'{method.title()} - Missing Values')

        plt.tight_layout()
        plt.show()