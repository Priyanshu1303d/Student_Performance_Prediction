import os 
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self , train_array , test_array):
        try:
            logging.info("Split Training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], # select all the rows and cols except last col
                train_array[:, -1], # select all the rows but only select the last col
                test_array[:, :-1],
                test_array[:, -1]

            )


            #   Creating dic of the models
            models = {
            "RandomForest": RandomForestRegressor(),
            "KNearestNeighbors": KNeighborsRegressor(),
            "DecisionTree": DecisionTreeRegressor(),
            "Xgboost": XGBRegressor(),
            "GradientBoosting": GradientBoostingRegressor(),
            "LinearRegression": LinearRegression(),
            "CatBoosting": CatBoostRegressor(verbose = False),
            "AdaBoosting": AdaBoostRegressor(),
            }

            model_report : dict = evaluate_models(X_train = X_train , X_test = X_test, y_train= y_train, y_test= y_test, models = models)


            # to get best model score 
            best_model_score = max(sorted(model_report.values()))

            #to get the best model name from the dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            
            if best_model_score < 0.6 :
                raise CustomException("No best Model found")
            
            logging.info(f"Best Model found on both training and testing dataset")

            #Saving the model
            save_object (
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            prediction = best_model.predict(X_test)

            r2_square = r2_score(y_test, prediction)

            return r2_square


        except Exception as e:
          raise CustomException(e , sys)

