from src.exception import CustomException
from src.logger import logger
from src.entity.config_entity import DataIngestionConfig

import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config = config
    
    def initiate_data_ingestion(self):
        logger.info("Entered the data ingestion class")
        
        try:
            df = pd.read_csv("notebook\data\house_price_regression_dataset.csv")
            logger.info("Read the dataset as dataframe")
            os.makedirs(os.path.dirname(self.config.raw_data_path),exist_ok=True)
            
            df.to_csv(self.config.raw_data_path,index=False)
            
            logger.info("Raw data is saved")
            logger.info("Train test split initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            
            train_set.to_csv(self.config.train_data_path,index=False)
            test_set.to_csv(self.config.test_data_path,index=False)
            logger.info("Train and test data saved")
            
            return (
                self.config.train_data_path,
                self.config.test_data_path
            )
        
        except Exception as e:
            logger.info("Exception occurred in data ingestion stage")
            raise CustomException(e,sys)
    