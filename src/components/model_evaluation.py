import os
import sys
import json
import joblib

from sklearn.metrics import(
    r2_score,
    mean_absolute_error
)

from src.exception import CustomException
from src.logger import logger
from src.entity.config_entity import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self,config:ModelEvaluationConfig):
        self.config = config
    
    def initiate_model_evaluation(self,test_arr,model_path):
        try:
            logger.info("Starting model evalutaion")
            x_test = test_arr[:,:-1]
            y_test = test_arr[:,-1]
            model = joblib.load(model_path)
            y_pred = model.predict(x_test)
            metrics = {
                "r2_score":r2_score(y_test,y_pred),
                "mean_absolute_error":mean_absolute_error(y_test,y_pred)
            }
            
            os.makedirs(os.path.dirname(self.config.metric_file_path),
                        exist_ok=True)
            
            with open(self.config.metric_file_path,"w") as f:
                json.dump(metrics,f,indent=4)
            logger.info("Evaluation metrics saved")
            return metrics
        except Exception as e:
            logger.info("Exception occurred in model evaluation stage")
            raise CustomException(e,sys)