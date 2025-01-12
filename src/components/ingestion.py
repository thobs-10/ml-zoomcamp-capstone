import os
import sys
from loguru import logger
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Union
from dotenv import load_dotenv

from src.components.component import Component
from src.exceptions import AppException

from src.entity.config import DataIngestionConfig, DataPreprocessing
from src.utils.main_utils import get_statistical_properties, get_outliers

# load_dotenv()

class DataPreprocessingComponent(Component):
    def __init__(self,
                data_source: DataIngestionConfig = DataIngestionConfig(),
                data_store: DataPreprocessing = DataPreprocessing()):
        
        self.data_source = data_source
        self.data_store = data_store

    def load_data(self) -> Union[pd.DataFrame, Tuple]:
        try:
            logger.info(f"Loading data from: {self.data_source.raw_data_path}")
            df = pd.read_csv(self.data_source.raw_data_path)
            df.drop(columns='Unnamed: 0', axis=1, inplace= True)
            return df
        except FileNotFoundError:
            raise AppException(f"File not found at: {self.data_source.raw_data_path}", sys)
        except Exception as e:
            raise AppException(f"Error loading data: {str(e)}", sys)
    
    def handling_categorical_types(self, df: pd.DataFrame)-> pd.DataFrame:
        logger.info("Handling categorical types ...")
        df['Gender'] = df['Gender'].astype('category')
        df['Customer Type'] = df['Customer Type'].astype('category')
        df['Type of Travel'] = df['Type of Travel'].astype('category')
        df['Class'] = df['Class'].astype('category')
        # df['delay_category'] = df['delay_category'].astype('category')
        logger.debug("Handled categorical types successfully")
        return df

    def handling_numeric_types(self, df:pd.DataFrame) -> pd.DataFrame:
        logger.info("Handling numeric types ...")
        for column in df.columns:
            if pd.api.types.is_object_dtype(df[column]):
                df[column] = df[column].astype(str)
            elif pd.api.types.is_numeric_dtype(df[column]):
                df[column] = pd.to_numeric(df[column], errors='coerce')
            elif pd.api.types.is_datetime64_dtype(df[column]):
                df[column] = pd.to_datetime(df[column])
        logger.debug("Handled numeric types successfully")
        return df

    def drop_duplicates(self, df:pd.DataFrame)-> pd.DataFrame:
        logger.info("Dropping duplicates ...")  
        duplicated = df[df.duplicated(keep=False)]
        if duplicated.shape[0] > 0:
            df = df.drop_duplicates(inplace= True)
        logger.debug("Dropped duplicates successfully")
        return df

    def handle_missing_values(self, df:pd.DataFrame) -> pd.DataFrame:
        logger.info("Handling missing values ...")
        features_with_na=[features for features in df.columns if df[features].isnull().sum()>=1]
        for feature in features_with_na:
            if df[feature].dtype == 'categorical':
                df[feature].fillna(df[feature].mode()[0], inplace=True)
            else:
                df[feature].fillna(df[feature].mean())
        logger.debug("Handled missing values successfully")
        return df

    def removing_outliers(self, df:pd.DataFrame) -> pd.DataFrame:
        logger.info("Removing outliers ...")
        for column in ['Age', 'Departure Delay in Minutes', 'Flight Distance']:
            Q1, Q3, IQR = get_statistical_properties(df, column)
            outlier = get_outliers(df, column, Q1, Q3, IQR)
            df = df[~outlier]
        logger.debug("Removed outliers successfully")
        return df
    
    def save(self, df:pd.DataFrame) -> None:
        try:
            os.makedirs("data/processed", exist_ok=True)
            logger.info(f"Saving data to: {self.data_store.processed_data_path}")
            df.to_csv(os.path.join(self.data_store.processed_data_path, "processed.csv"), index=False)
        except Exception as e:
            raise AppException(f"Error saving data: {str(e)}", sys)


