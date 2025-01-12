from dataclasses import dataclass, field
from typing import List, Tuple, Union
import os
from dotenv import load_dotenv

# root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

@dataclass
class DataIngestionConfig:
    raw_data_path : str = os.getenv('RAW_PATH_FILE')

@dataclass
class DataPreprocessing:
    processed_data_path: str = os.getenv('PROCESSED_PATH_FILE')

@dataclass
class ArtifactStore:
    artifacts_path: str = os.getenv('ARTIFACTS_PATH')
@dataclass
class FeatureEngineeringConfig:
    feature_engineered_data_path: str =os.getenv('FEATURE_ENGINEERED_DATA_PATH')
    
@dataclass
class ModelTrainingConfig:
    artifact_output_path : str = os.getenv('MODEL_ARTIFACT_PATH')
    model_output_path : str = os.getenv('MODEL_OUTPUT_PATH')
    
@dataclass
class ModelTuningConfig:
    hyperparameters : dict = field(default_factory=lambda:{
    "max_depth": [1, 3, 6, 8, 10],
    "n_estimators": [50, 100, 150, 250, 300], 
    "min_samples_split": [2, 4, 6, 8, 10],
    "min_samples_leaf": [1, 2, 3, 4, 5],
    })
    model_pipeline_path : str = os.getenv('MODEL_PIPELINE_PATH')