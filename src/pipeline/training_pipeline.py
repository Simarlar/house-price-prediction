from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig
from src.components.data_validation import DataValidation
from src.entity.config_entity import DataValidationConfig

if __name__ == "__main__":
    
    # Data Ingestion
    data_ingestion_config = DataIngestionConfig(
        raw_data_path="artifacts/raw_data.csv",
        train_data_path="artifacts/train_data.csv",
        test_data_path="artifacts/test_data.csv"
    )
    ingestion = DataIngestion(config=data_ingestion_config)
    train_path,test_path = ingestion.initiate_data_ingestion()
    
    # Data Validation
    data_validation_config = DataValidationConfig(
        unzip_data_dir="artifacts/raw_data.csv",
        STATUS_FILE="artifacts/validation_status.txt"
    )
    
    validation = DataValidation(config=data_validation_config)
    validation.validate_all_columns()
    