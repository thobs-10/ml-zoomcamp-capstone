import pytest
import pandas as pd
import os
from unittest.mock import MagicMock, patch
from src.components.ingestion import DataPreprocessingComponent, DataIngestionConfig, DataPreprocessing, AppException

@pytest.fixture
def data_preprocessor():
    data_source = MagicMock(spec=DataIngestionConfig)
    data_store = MagicMock(spec=DataPreprocessing)
    data_source.raw_data_path = "test_data.csv"
    data_store.processed_data_path = "data/processed"
    return DataPreprocessingComponent(data_source=data_source, data_store=data_store)

@pytest.fixture
def sample_dataframe():
    data = {
        "Unnamed: 0": [0, 1],
        "Gender": ["Male", "Female"],
        "Customer Type": ["Loyal Customer", "Disloyal Customer"],
        "Type of Travel": ["Business travel", "Personal Travel"],
        "Class": ["Business", "Economy"],
        "Age": [25, 30],
        "Departure Delay in Minutes": [5, 10],
        "Flight Distance": [200, 400],
    }
    return pd.DataFrame(data)

def test_load_data_success(data_preprocessor, sample_dataframe, tmp_path):
    file_path = tmp_path / "test_data.csv"
    sample_dataframe.to_csv(file_path, index=False)
    data_preprocessor.data_source.raw_data_path = str(file_path)

    df = data_preprocessor.load_data()
    assert not df.empty
    assert "Unnamed: 0" not in df.columns

def test_load_data_file_not_found(data_preprocessor):
    data_preprocessor.data_source.raw_data_path = "non_existent.csv"
    with pytest.raises(AppException):
        data_preprocessor.load_data()

def test_handling_categorical_types(data_preprocessor, sample_dataframe):
    df = data_preprocessor.handling_categorical_types(sample_dataframe)
    assert df["Gender"].dtype.name == "category"
    assert df["Customer Type"].dtype.name == "category"

def test_handling_numeric_types(data_preprocessor, sample_dataframe):
    df = data_preprocessor.handling_numeric_types(sample_dataframe)
    assert pd.api.types.is_numeric_dtype(df["Age"])
    assert pd.api.types.is_numeric_dtype(df["Departure Delay in Minutes"])


def test_save(data_preprocessor, sample_dataframe, tmp_path):
    output_dir = tmp_path / "data/processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_preprocessor.data_store.processed_data_path = str(output_dir)

    data_preprocessor.save(sample_dataframe)
    saved_file = output_dir / "processed.csv"
    assert saved_file.exists()

