import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Any, Union
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.feature_selection import mutual_info_classif, chi2, f_classif
from sklearn.ensemble import RandomForestClassifier
from loguru import logger
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator
from typing import Union, Optional, Tuple, Any
from sklearn.pipeline import Pipeline, make_pipeline
import warnings
warnings.filterwarnings('ignore')



class CategoricalFeatureSelector:
    """
    Feature selector for classification problems with mixed categorical and numerical features.
    Combines multiple selection methods:
    1. Mutual Information for non-linear relationships
    2. Chi-Square for categorical relationships
    3. ANOVA F-value for numerical features
    4. Random Forest importance for complex interactions
    """
    
    def __init__(self, n_features: Optional[int]=10):
        self.n_features = n_features
        self.feature_scores = None
        self.selected_features = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def _preprocess_data(self, X):
        """
        Preprocess data by handling categorical variables, infinities, and scaling
        """
        X_processed = X.copy()
        categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            X_processed[col] = self.label_encoders[col].fit_transform(X_processed[col].astype(str))

        numerical_cols = X_processed.select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_cols) > 0:
            X_processed[col] = X_processed[col].replace([np.inf, -np.inf], np.nan)
            medians = X_processed[col].median()
            quantiles = X_processed[numerical_cols].quantile([0.01, 0.99])
            X_processed[numerical_cols] = X_processed[numerical_cols].fillna(medians)
            X_processed[numerical_cols] = X_processed[numerical_cols].clip(quantiles.loc[0.01], quantiles.loc[0.99], axis =1)
            X_processed[numerical_cols] = self.scaler.fit_transform(X_processed[numerical_cols])
        
        return X_processed
        
    def fit(self, X, y):
        """
        Fit the feature selector to the data
        
        Parameters:
        -----------
        X : pandas DataFrame
            Input features (mixed types)
        y : array-like
            Target variable (categorical)
        """
        X_processed = self._preprocess_data(X)
        if not isinstance(y, (np.ndarray, pd.Series)):
            y = np.array(y)
        if y.dtype == object or isinstance(y.dtype, pd.CategoricalDtype):
            y = LabelEncoder().fit_transform(y)

        scores_dict = {}
        scores_dict = self._calculate_scores(X_processed, y)
       
            
        # Combine scores using weighted average
        weights = {
            'mutual_info': 0.3,    # Good for non-linear relationships
            'chi2': 0.2,          # Good for categorical features
            'f_value': 0.2,       # Good for numerical features
            'random_forest': 0.3   # Good for interactions
        }
        normalized_scores = {method: self._normalize_score(scores) for method, scores in scores_dict.items()}
        final_scores = {}
        final_scores = {
            feature : sum(weights[method] * normalized_scores[method].get(feature, 0.0) for method in weights)
            for feature in X.columns
        }
            
        self.feature_scores = final_scores
        self.selected_features = sorted(final_scores.items(), 
                                      key=lambda x: x[1], 
                                      reverse=True)[:self.n_features]
        
        return self
    
    def transform(self, X):
        """Return dataset with only selected features"""
        selected_feature_names = [feature[0] for feature in self.selected_features]
        return X[selected_feature_names]
    
    def fit_transform(self, X, y):
        """Fit and transform the data"""
        return self.fit(X, y).transform(X)
    
    def get_feature_importance(self, include_scores=True):
        """
        Return feature importance scores and ranks
        
        Parameters:
        -----------
        include_scores : bool
            If True, includes normalized scores for each selection method
        """
        scores_df = pd.DataFrame(self.selected_features, 
                               columns=['Feature', 'Combined_Score'])
        scores_df['Rank'] = range(1, len(scores_df) + 1)
        
        if include_scores:
            scores_df = scores_df.round(4)
            
        return scores_df
    
    def _calculate_scores(self, X_processed: pd.DataFrame, y: pd.Series) -> dict[str, Any]:

        def compute_mutual_info():
            return dict(zip(
                X_processed.columns,
                mutual_info_classif(X_processed, y, random_state=42)
            ))
        
        def compute_chi2():
            x_chi = X_processed - X_processed.min() + 0.1
            chi_scores, _ = chi2(x_chi, y)
            return dict(zip(X_processed.columns, chi_scores))
        
        def compute_f_value():
            f_scores, _ = f_classif(X_processed, y)
            return dict(zip(X_processed.columns, f_scores))
        
        def compute_random_forest():
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_processed, y)
            return dict(zip(X_processed.columns, rf_model.feature_importances_))
        
        results = Parallel(n_jobs=2)(delayed(fn)() for fn in [compute_mutual_info, compute_chi2, compute_f_value, compute_random_forest])
        return dict(zip(['mutual_info', 'chi2', 'f_value', 'random_forest'], results))
    
    @staticmethod
    def _normalize_score(scores) -> Union[dict[Any, float], dict]:
        """Normalize score to [0, 1] range with error handling"""
    
        scores = np.array(list(scores.values()))
        min_val = np.nanmin(scores)
        max_val = np.nanmax(scores)
        if min_val == max_val:
            return {feature: 0.0 for feature in scores.keys()}
        normalized = (scores - min_val) / (max_val - min_val + 1e-10)
        return dict(zip(scores, normalized))

def get_statistical_properties(df:pd.DataFrame, column: str) -> Tuple[float, float, float]:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return Q1, Q3, IQR

def get_outliers(df: pd.DataFrame, column: str, Q1: int | float,
                 Q3: int | float,
                 IQR: int | float) -> pd.Series:
    """
    Identify outliers based on the Interquartile Range (IQR) rule.
    """
    outlier = (df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)
    return outlier

def get_ensemble_models() -> list:
        """Return a list of ensemble models."""
        return [
            RandomForestClassifier(n_estimators=100, random_state=42),
            GradientBoostingClassifier(n_estimators=100, random_state=42),
            # CatBoostClassifier(iterations=100, random_state=42),
            # XGBClassifier(n_estimators=100, random_state=42),
            LGBMClassifier(n_estimators=100, random_state=42)
        ]

def convert_to_series(y_train: pd.DataFrame, y_valid: pd.DataFrame)-> Tuple[Union[pd.Series, pd.DataFrame],
                                                                            Union[pd.Series, pd.DataFrame]]:
    """Convert DataFrame columns to Series for compatibility with scikit-learn."""
    y_train = y_train.reset_index(drop=True).squeeze()
    y_valid = y_valid.reset_index(drop=True).squeeze()
    return y_train, y_valid

def create_best_model_pipeline(model: BaseEstimator, scaler: StandardScaler) -> Pipeline:
    """Create a pipeline with the selected model and scaler."""
    return Pipeline([
        ('scaler', scaler),
        ('model', model)
    ])