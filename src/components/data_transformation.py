import os
import pandas as pd
import numpy as np
import joblib
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from src.logger import logger
from src.exception import CustomException
from src.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self,config:DataTransformationConfig):
        self.config = config
    
    def get_data_transfomer_object(self):
        try:
            numerical_columns = ['Square_Footage',
                                 'Num_Bedrooms', 
                                 'Num_Bathrooms',
                                 'Year_Built',
                                'Lot_Size', 
                                'Garage_Size', 
                                'Neighborhood_Quality', 
                                'House_Age', 
                                'SqFt_Per_Bedroom', 
                                'House_Density', 
                                'Luxury_Score']
            
            num_pipeline = Pipeline(steps=
                                    [
                                        ('imputer',SimpleImputer(strategy='median')),
                                        ('scaler',StandardScaler())
                                    ])
            preprocessor = ColumnTransformer(transformers=[
                ('num_pipeline',num_pipeline,numerical_columns)
                
            ])
            return preprocessor
        
        except Exception as e:
            logger.info("Error in get data transformer object")
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logger.info("Read train and test data ")
            target_column = "House_Price"
            train_df["House_Age"] = 2026 - train_df["Year_Built"]

            train_df["SqFt_Per_Bedroom"] = (
                train_df["Square_Footage"] / (train_df["Num_Bedrooms"] + 1)
            )

            train_df["House_Density"] = (
                train_df["Square_Footage"] / (train_df["Lot_Size"] + 1)
            )

            train_df["Luxury_Score"] = (
                train_df["Square_Footage"] * train_df["Neighborhood_Quality"]
            )
            
            test_df["House_Age"] = 2026 - test_df["Year_Built"]

            test_df["SqFt_Per_Bedroom"] = (
                test_df["Square_Footage"] / (test_df["Num_Bedrooms"] + 1)
            )

            test_df["House_Density"] = (
                test_df["Square_Footage"] / (test_df["Lot_Size"] + 1)
            )

            test_df["Luxury_Score"] = (
                test_df["Square_Footage"] * test_df["Neighborhood_Quality"]
            )
            
            x_train = train_df.drop(columns=target_column,axis=1)
            y_train = train_df[target_column]
            
            x_test = test_df.drop(columns=target_column,axis=1)
            y_test = test_df[target_column]
            
            preprocessor_obj = self.get_data_transfomer_object()
            
            x_train_transformed = preprocessor_obj.fit_transform(x_train)
            x_test_transformed = preprocessor_obj.transform(x_test)
            
            logger.info("Applied preprocessing object on train and test data")
            
            train_arr = np.c_[x_train_transformed,np.array(y_train)]
            test_arr = np.c_[x_test_transformed,np.array(y_test)]
            
            os.makedirs(os.path.dirname(self.config.preprocessor_obj_file_path),exist_ok=True)
            joblib.dump(preprocessor_obj,self.config.preprocessor_obj_file_path)
            logger.info("Saved preprocessor object")
            
            return(
                train_arr,
                test_arr,
                self.config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
            
            
