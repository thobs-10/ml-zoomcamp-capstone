from src.exceptions import AppException

from src.components.feature_engineering import FeatureEngineeringComponent
from src.entity.config import DataPreprocessing, FeatureEngineeringConfig

class FeatureEngineeringPipeline:
    def __init__(self,
                data_source: DataPreprocessing = DataPreprocessing(),
                data_destination: FeatureEngineeringConfig = FeatureEngineeringConfig()):
        
        self.load_data_source = data_source
        self.save_data_destination = data_destination
    
    def run_feature_engineering(self)-> None:
        feature_engineering = FeatureEngineeringComponent(self.load_data_source, self.save_data_destination)
        df = feature_engineering.load_data()
        df = feature_engineering.feature_extraction(df)
        df = feature_engineering.encode_categorical_features(df)
        X, y = feature_engineering.drop_highly_correlated_features(df)
        features, y = feature_engineering.feature_selection(X, y)
        X_train, X_val, y_train, y_val = feature_engineering.split_dataset(features, y)
        feature_engineering.save_features(features, y)
        feature_engineering.save_data_splits(X_train, y_train, X_val, y_val)

if __name__ == "__main__":
    feature_engineering_pipeline = FeatureEngineeringPipeline()
    feature_engineering_pipeline.run_feature_engineering()
   