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

            # Hyperparameter Tuning
            params={
                "DecisionTree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "RandomForest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "GradientBoosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "LinearRegression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoostingRegressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoostRegressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report : dict = evaluate_models(X_train = X_train , X_test = X_test, y_train= y_train, y_test= y_test, models = models , param = params)


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

