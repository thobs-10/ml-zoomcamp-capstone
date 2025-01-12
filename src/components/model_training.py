import os
import sys
from loguru import logger
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Union
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline, Pipeline
import joblib

from src.components.component import Component
from src.exceptions import AppException
from src.entity.config import FeatureEngineeringConfig, ModelTrainingConfig
from src.utils.main_utils import get_ensemble_models, convert_to_series

class ModelTrainingComponent(Component):
    def __init__(self,
                feature_engineering_config: FeatureEngineeringConfig = FeatureEngineeringConfig(),
                model_training_config: ModelTrainingConfig = ModelTrainingConfig()):
        
        self.features_config = feature_engineering_config
        self.model_data_path = model_training_config
        self.models = []
        self.model_metrics ={}

    def load_data(self) -> Union[pd.DataFrame, Tuple]:
        try:
            logger.info(f"Loading data from: {self.features_config.feature_engineered_data_path}")
            X_train = pd.read_csv(os.path.join(self.features_config.feature_engineered_data_path, "X_train.csv"))
            X_val = pd.read_csv(os.path.join(self.features_config.feature_engineered_data_path, "X_val.csv"))
            y_train = pd.read_csv(os.path.join(self.features_config.feature_engineered_data_path, "y_train.csv"))
            y_val = pd.read_csv(os.path.join(self.features_config.feature_engineered_data_path, "y_valid.csv"))
            y_train, y_val = convert_to_series(y_train, y_val)
            logger.debug("Data loaded successfully.")
            return X_train, y_train, X_val, y_val
        except FileNotFoundError:
            raise AppException(f"File not found at: {self.load_data.feature_engineered_data_path}", sys)
        except Exception as e:
            raise AppException(f"Error loading data: {str(e)}", sys)

    def feature_scaling(self, X_train: pd.DataFrame, X_valid: pd.DataFrame) -> Tuple[
                                                                        Union[pd.DataFrame, np.ndarray],
                                                                        Union[pd.DataFrame, np.ndarray],
                                                                        StandardScaler, list[str]]:
        
        logger.info("Feature scaling...")
        excluded_features = ['Type of Travel', 'Class', 'Customer Type', 'delay_category', 'has_delay']
        X_train.drop(columns='id', axis=1, inplace=True)
        X_valid.drop(columns='id', axis=1, inplace=True)
        scaling_cols = [col for col in X_train.columns if col not in excluded_features]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train[scaling_cols])
        X_valid_scaled = scaler.transform(X_valid[scaling_cols])
        logger.debug("Feature scaling completed.")
        return X_train_scaled, X_valid_scaled, scaler, scaling_cols

    def train_models(self, X_train_scaled: np.ndarray, y_train: pd.DataFrame) -> None:
        logger.info("Training models...")
        for model in get_ensemble_models():
            model.fit(X_train_scaled, y_train)
            self.models.append(model)
        logger.debug("Training completed sucesssfully")

    def evaluate_model(self, X_val_scaled: np.ndarray, y_val: pd.Series, scaler: StandardScaler, scaling_columns: list[str]) -> None:
        logger.info("Evaluating model...")
        for model in self.models:
            y_pred = model.predict(X_val_scaled)
            self.model_metrics[model.__class__.__name__] = accuracy_score(y_val, y_pred)
        logger.debug("Evaluation completed successfully.")

    def select_best_model(self) -> Union[BaseEstimator, Pipeline]:
        logger.info("Selecting best model...")
        best_model = max(zip(self.model_metrics.values(), self.model_metrics.keys()))[1]
        wanted_model = [m for m in self.models if m.__class__.__name__ == best_model][0]
        logger.debug("Best model selected successfully.")
        return wanted_model
    
    def save_model(self, model: BaseEstimator, scaler: StandardScaler) -> None:
        try:
            logger.info(f"Saving model to: {self.model_data_path}")
            model_path = os.path.join(self.model_data_path.model_output_path, "model.joblib")
            scaler_path = os.path.join(self.model_data_path.artifact_output_path, "scaler.joblib")
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            logger.debug("Model and artifact saved successfully.")
        except Exception as e:
            raise AppException(f"Error saving model: {str(e)}", sys)
        