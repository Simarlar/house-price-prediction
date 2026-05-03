import os
import sys
import joblib
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from src.logger import logger
from src.exception import CustomException
from src.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self,config:ModelTrainerConfig):
        self.config = config
    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logger.info("Splitting training and testing dataset in model trainer stage")
            x_train,y_train = train_arr[:,:-1], train_arr[:,-1]
            x_test,y_test = test_arr[:,:-1],test_arr[:,-1]
            
            models = {
                "Linear Regression" : LinearRegression(),
                "Random Forest Regressor":RandomForestRegressor()
            }
            
            model_report = {}
            mlflow.set_experiment("House Price Prediction")
            best_model_name = None
            best_r2_score = -1
            for model_name,model in models.items():
                with mlflow.start_run(run_name=model_name):
                    logger.info(f"Traning {model_name}")
                    model.fit(x_train,y_train)
                    y_pred = model.predict(x_test)
                    
                    score = r2_score(y_pred,y_test)
                    
                    mlflow.log_param("model_name",model_name)
                    mlflow.log_metric("r2_score",score)
                    
                    mlflow.sklearn.log_model(
                        model,
                        artifact_path = "model",
                        registered_model_name = "HousePricePrediction"
                    )
                    
                    model_report[model_name] = {
                        "r2_score":score
                    }
                    
                    if score > best_r2_score:
                        best_r2_score = score
                        best_model_name = model_name
                
            best_model = models[best_model_name]
            os.makedirs(
                os.path.dirname(self.config.trained_model_file_path),
                exist_ok=True
            )
            joblib.dump(best_model,self.config.trained_model_file_path)
            logger.info("Best model saved successfully")
            return model_report
        
        except Exception as e:
            logger.info("Exception occurred in the model trainer stage")
            raise CustomException(e,sys)
        