from src.exceptions import AppException

from src.components.ingestion import DataPreprocessingComponent
from src.entity.config import DataIngestionConfig, DataPreprocessing


class DataIngestionPipeline:
    def __init__(self,
                data_source: DataIngestionConfig = DataIngestionConfig(),
                data_destination: DataPreprocessing = DataPreprocessing(),):
        self.data_source = data_source
        self.data_destination = data_destination

    def run_data_ingestion(self):
        data_preprocessing = DataPreprocessingComponent(self.data_source, self.data_destination)
        df = data_preprocessing.load_data()
        df = data_preprocessing.handling_categorical_types(df)
        df = data_preprocessing.handling_numeric_types(df)
        df = data_preprocessing.drop_duplicates(df)
        df = data_preprocessing.handle_missing_values(df)
        df = data_preprocessing.removing_outliers(df)
        data_preprocessing.save(df)

if __name__ == "__main__":
    pipeline = DataIngestionPipeline()
    pipeline.run_data_ingestion()
