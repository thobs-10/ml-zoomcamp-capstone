from src.components.model_tuning import ModelTuningComponent
from src.pipelines.data_ingestion_pipeline import DataIngestionPipeline
from src.pipelines.feature_engineering_pipeline import FeatureEngineeringPipeline
from src.pipelines.model_training_pipeline import ModelTrainingPipeline

def run_data_ingestion_pipeline():
    data_ingestion_pipeline = DataIngestionPipeline()
    data_ingestion_pipeline.run_data_ingestion()

def run_feature_engineering_pipeline():
    feature_engineering_pipeline = FeatureEngineeringPipeline()
    feature_engineering_pipeline.run_feature_engineering()

def run_model_training_pipeline():
    model_tuning = ModelTuningComponent()  # Load hyperparameters from a configuration file or environment variables
    model_training_pipeline = ModelTrainingPipeline(model_tuning=model_tuning)
    model_training_pipeline.run_model_training()

if __name__ == "__main__":
    run_data_ingestion_pipeline()
    run_feature_engineering_pipeline()
    run_model_training_pipeline()