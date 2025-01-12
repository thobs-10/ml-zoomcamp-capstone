import os
import sys
from loguru import logger
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Union
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from src.components.component import Component
from src.exceptions import AppException
from src.utils.main_utils import CategoricalFeatureSelector
from src.entity.config import DataPreprocessing, FeatureEngineeringConfig

class FeatureEngineeringComponent(Component):
    def __init__(self,
                data_source: DataPreprocessing = DataPreprocessing(),
                data_destination: FeatureEngineeringConfig = FeatureEngineeringConfig()):
        
        self.load_data_source = data_source
        self.save_data_destination = data_destination

    def load_data(self) -> Union[pd.DataFrame, Tuple]:
        try:
            logger.info(f"Loading data from: {self.load_data_source.processed_data_path}")
            df = pd.read_csv(os.path.join(self.load_data_source.processed_data_path, "processed.csv"))
            return df
        except FileNotFoundError:
            raise AppException(f"File not found at: {self.load_data_source.raw_data_path}", sys)
        except Exception as e:
            raise AppException(f"Error loading data: {str(e)}", sys)
    
    def feature_extraction(self, df: pd.DataFrame) -> pd.DataFrame:
        df['average_service_score'] = (df['Inflight wifi service'] + df['Food and drink'] + \
                                    df['Seat comfort'] + df['Inflight entertainment'] + df['Cleanliness'] + \
                                    df['Inflight service'] + df['Checkin service'] + \
                                    df['Leg room service']) / 8

        df['total_delay'] = df['Departure Delay in Minutes'] + df['Arrival Delay in Minutes']
        df['average_delay_time'] = (df['Departure Delay in Minutes'] + df['Arrival Delay in Minutes']) / 2
        df['delay_category'] = df['average_delay_time'].apply(lambda x: 'No Delay' if x <= 25 else ('Short Delay' if x <= 50 else ('Moderate Delay' if x <= 75 else 'Severe Delay')))
        df['has_delay'] = df['total_delay'].apply(lambda x: 1 if x> 0.0 else 0)
        return df

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Gender'] = np.where(df['Gender'] == 'Male', 1, 0)
        df['Customer Type'] = np.where(df['Customer Type'] == 'Loyal Customer', 1, 0)
        df['Type of Travel'] = np.where(df['Type of Travel'] == 'Business Travel', 1, 0)
        label_encode_class = { value: key for key, value in enumerate(df['Class'].unique())}
        df['Class'] = df['Class'].map(label_encode_class)
        label_encode_delay_category = {value: key for key, value in enumerate(df['delay_category'].unique())}
        df['delay_category'] = df['delay_category'].map(label_encode_delay_category)
        return df

    def separate_dataset(self, df: pd.DataFrame)-> Tuple[pd.DataFrame, pd.Series]:
        X = df.drop(columns='satisfaction', axis= 1)
        y = df['satisfaction']
        return X, y

    def drop_highly_correlated_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        X, y  = self.separate_dataset(df)
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        highly_corr_features = [column for column in upper.columns if any(upper[column] > 0.95)]
        df.drop(columns=highly_corr_features, axis=1, inplace=True)
        return df, y

    def encode_target_feature(self, y: pd.Series) -> pd.Series:
        label_encoder = LabelEncoder()
        y  = label_encoder.fit_transform(y)
        return y

    def feature_selection(self, df: pd.DataFrame, y: pd.Series)-> Tuple[pd.DataFrame, pd.Series]:
        y = self.encode_target_feature(y)
        selector = CategoricalFeatureSelector(n_features=10)
        X_selected = selector.fit_transform(df, y)
        return X_selected, y
    
    def split_dataset(self, df: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame,
                                                           pd.Series, pd.Series]:
        
        X_train, X_val, y_train, y_val = train_test_split(df, y,
                                                            test_size=0.25, random_state=42)
        return X_train, X_val, y_train, y_val
    
    def save_features(self, X_selected: pd.DataFrame, y: pd.Series) -> None:
        try:
            os.makedirs("data/features", exist_ok=True)
            logger.info(f"Saving data to: {self.save_data_destination.feature_engineered_data_path}")
            X_selected.to_csv(os.path.join(self.save_data_destination.feature_engineered_data_path, "features.csv"), index=False)
            pd.DataFrame(y).to_csv(os.path.join(self.save_data_destination.feature_engineered_data_path, "target.csv"), index=False)

        except Exception as e:
            raise AppException(f"Error saving data: {str(e)}", sys)
        
    def save_data_splits(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_valid: pd.DataFrame, y_valid: pd.Series)-> None:
        """save the splits in the same data location as the above features and target"""
        try:
            logger.info(f"Saving data splits to: {self.save_data_destination.feature_engineered_data_path}")
            X_train.to_csv(os.path.join(self.save_data_destination.feature_engineered_data_path, "X_train.csv"), index=False)
            X_valid.to_csv(os.path.join(self.save_data_destination.feature_engineered_data_path, "X_val.csv"), index=False)
            pd.DataFrame(y_train).to_csv(os.path.join(self.save_data_destination.feature_engineered_data_path, "y_train.csv"), index=False)
            pd.DataFrame(y_valid).to_csv(os.path.join(self.save_data_destination.feature_engineered_data_path, "y_valid.csv"), index=False)
            logger.debug("saving data done successfully")
        except Exception as e:
            raise AppException(f"Error saving data splits: {str(e)}", sys)