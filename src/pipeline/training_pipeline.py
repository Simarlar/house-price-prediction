from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig
from src.components.data_validation import DataValidation
from src.entity.config_entity import DataValidationConfig
from src.components.data_transformation import DataTransformation
from src.entity.config_entity import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.entity.config_entity import ModelTrainerConfig
from src.entity.config_entity import ModelEvaluationConfig
from src.components.model_evaluation import ModelEvaluation
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
    
    # Data Transformation
    transformation_config = DataTransformationConfig(
        preprocessor_obj_file_path="artifacts/preprocessor.pkl"
    )
    
    transformation = DataTransformation(transformation_config)
    train_arr , test_arr, _ = transformation.initiate_data_transformation(
        train_path=train_path,
        test_path=test_path
    )
    
    # Model Trainer
    
    model_trainer_config = ModelTrainerConfig(trained_model_file_path="artifacts/model.pkl")
    trainer = ModelTrainer(model_trainer_config)
    model_report = trainer.initiate_model_trainer(
        train_arr=train_arr,
        test_arr=test_arr
    )
    # Model Trainer
    model_evaluation_config = ModelEvaluationConfig(
        metric_file_path="artifacts/metric_file.json"
    )
    
    model_evaluation = ModelEvaluation(model_evaluation_config)
    metrics = model_evaluation.initiate_model_evaluation(
        test_arr=test_arr,
        model_path="artifacts/model.pkl"
    )
    print(metrics)
    