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
from sklearn.model_selection import  RandomizedSearchCV, KFold, cross_val_score
import joblib
from joblib import Parallel, delayed
import multiprocessing as mp

from src.components.component import Component
from src.exceptions import AppException
from src.entity.config import ModelTrainingConfig, ModelTuningConfig
from src.utils.main_utils import create_best_model_pipeline

import warnings
warnings.filterwarnings('ignore')

class ModelTuningComponent:
    def __init__(self,
                model_training_config: ModelTrainingConfig = ModelTrainingConfig(),
                model_tuning_config: ModelTuningConfig = ModelTuningConfig()):
        
        self.model_training_config = model_training_config
        self.hyperparameters = model_tuning_config.hyperparameters
        self.model_tuning_config = model_tuning_config
        
    
    def load_model(self) -> Tuple[BaseEstimator, StandardScaler]:
        try:
            logger.info(f"Loading model from: {self.model_training_config}")
            model = joblib.load(os.path.join(self.model_training_config.model_output_path, "model.joblib"))
            scaler = joblib.load(os.path.join(self.model_training_config.artifact_output_path, "scaler.joblib"))
            logger.debug("Model loaded successfully.")
            return model, scaler
        except FileNotFoundError:
            raise AppException(f"File not found at: {self.model_training_config}", sys)
        except Exception as e:
            raise AppException(f"Error loading model: {str(e)}", sys)
    
    def tune_hyperparameters(self, model: BaseEstimator, X_train_scaled: np.ndarray, y_train: pd.Series) -> RandomizedSearchCV:
        logger.info("Performing hyperparameter tuning...")
        random_cv_model = RandomizedSearchCV(
                                    estimator=model,
                                    param_distributions=self.hyperparameters,
                                    n_iter=10,
                                    cv=5,
                                    verbose=2,
                                    n_jobs=-1,
                                    random_state=42)
        
        random_cv_model.fit(X_train_scaled, y_train)
        logger.info("Hyperparameter tuning succeeded")
        return random_cv_model
    
    # def perform_cross_validation(self, model: BaseEstimator, X_valid_scaled: np.ndarray, y_val: pd.Series) -> BaseEstimator:
    #     logger.info("Performing cross-validation...")
    #     kf = KFold(n_splits=5)
    #     # parallization for efficiency of the code
    #     def evaluate_folds():
    #         best_model = model.best_estimator_
    #         scores = cross_val_score(best_model, X_valid_scaled, y_val, cv=kf)
    #         return scores, best_model
    #     scores, best_model = Parallel(n_jobs=mp.cpu_count())(delayed(evaluate_folds)() for _ in range(10))
    #     logger.info(f"Cross-validation score: {scores.mean()}")
    #     if scores.mean()<0.80:
    #         raise ValueError("Cross validation score is below the threshold")
    #     logger.info("Cross-validation succeeded")
    #     return best_model
    

    def perform_cross_validation(self, model: BaseEstimator, X_valid_scaled: np.ndarray, y_val: pd.Series) -> BaseEstimator:
        """
        Perform cross-validation on the given model using the validation data.
        
        Args:
            model (BaseEstimator): The model to evaluate.
            X_valid_scaled (np.ndarray): Scaled validation features.
            y_val (np.ndarray): Validation target values.

        Returns:
            BaseEstimator: The best estimator after cross-validation.

        Raises:
            ValueError: If the cross-validation score is below the threshold.
        """
        logger.info("Performing cross-validation...")
        kf = KFold(n_splits=5)
        # best_model = model

        # Parallelize cross-validation across CPU cores
        scores = cross_val_score(model, X_valid_scaled, y_val, cv=kf, n_jobs=mp.cpu_count())
        mean_score = scores.mean()
        logger.info(f"Cross-validation scores: {scores}")
        logger.info(f"Mean cross-validation score: {mean_score}")
        if mean_score < 0.80:
            raise ValueError("Cross-validation score is below the threshold")

        logger.info("Cross-validation succeeded")
        return model
    
    
    def save_best_model(self, model: BaseEstimator, scaler: StandardScaler)-> None:
        try:
            os.makedirs("models/best_model", exist_ok=True)
            model_pipeline = create_best_model_pipeline(model, scaler)
            logger.info(f"Saving the best model to: {self.model_tuning_config.model_pipeline_path}")
            joblib.dump(model_pipeline, os.path.join(self.model_tuning_config.model_pipeline_path, "model_pipeline.joblib"))
            logger.debug("Best model saved successfully.")
        except Exception as e:
            raise AppException(f"Error saving best model: {str(e)}", sys)
