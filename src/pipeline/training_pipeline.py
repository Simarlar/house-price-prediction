from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig

if __name__ == "__main__":
    data_ingestion_config = DataIngestionConfig(
        raw_data_path="artifacts/raw_data.csv",
        train_data_path="artifacts/train_data.csv",
        test_data_path="artifacts/test_data.csv"
    )
    ingestion = DataIngestion(config=data_ingestion_config)
    train_path,test_path = ingestion.initiate_data_ingestion()
    