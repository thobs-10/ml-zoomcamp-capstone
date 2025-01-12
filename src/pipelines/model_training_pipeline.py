from src.exceptions import AppException

from src.components.model_training import ModelTrainingComponent
from src.components.model_tuning import ModelTuningComponent
from src.entity.config import FeatureEngineeringConfig, ModelTrainingConfig, ModelTuningConfig

class ModelTrainingPipeline:
    def __init__(self,
                feature_engineering_config: FeatureEngineeringConfig = FeatureEngineeringConfig(),
                model_training_config: ModelTrainingConfig = ModelTrainingConfig(),
                model_tuning: ModelTuningComponent = ModelTuningComponent):
        
        self.features_data_config = feature_engineering_config
        self.model_data_config = model_training_config
        self.model_tuning = model_tuning
    
    def run_model_training(self)-> None:
        model_training = ModelTrainingComponent(self.features_data_config, self.model_data_config)
        X_train, y_train, X_val, y_val = model_training.load_data()
        X_train_scaled, X_valid_scaled, scaler, scaling_cols = model_training.feature_scaling(X_train, X_val)
        model_training.train_models(X_train_scaled, y_train)
        model_training.evaluate_model(X_valid_scaled, y_val, scaler, scaling_cols)
        best_model = model_training.select_best_model()
        model_training.save_model(best_model, scaler)

        # model_tuning = ModelTuningComponent(self.model_data_config, self.model_tuning_config)
        model, scaler = self.model_tuning.load_model()
        random_cv_model = self.model_tuning.tune_hyperparameters(model, X_train_scaled, y_train)
        best_model = self.model_tuning.perform_cross_validation(random_cv_model.best_estimator_, X_valid_scaled, y_val)
        self.model_tuning.save_best_model(best_model, scaler)



if __name__ == "__main__":
    model_tuning = ModelTuningComponent()
    pipeline = ModelTrainingPipeline(model_tuning= model_tuning)
    pipeline.run_model_training()
