import os
import sys
import pandas as pd
import yaml

from src.exception import CustomException
from src.logger import logger
from src.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self,config:DataValidationConfig):
        self.config = config
    
    def validate_all_columns(self):
        logger.info("Entered the data validation class")
        try:
            validation_status = True
            data = pd.read_csv(self.config.unzip_data_dir)
            with open("config/schema.yaml") as f:
                schema = yaml.safe_load(f)
            
            all_cols = schema["columns"]
            
            for col in all_cols:
                if col not in data.columns:
                    validation_status = False
            
            with open(self.config.STATUS_FILE,"w") as f:
                f.write(f"Validation status: {validation_status}")
            
            return validation_status
        
        except Exception as e:
            logger.info("Exception occurred in data validation stage")
            raise CustomException(e,sys)
    