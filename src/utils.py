import os 
import sys

import pandas as pd
import numpy as np
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException


def save_object(file_path , obj):
    try:
        #saving the file path 
        dir_name = os.path.dirname(file_path)
        #makes the directory if it doesn't exist
        os.makedirs(dir_name , exist_ok= True)

        #writing in file
        with open(file_path , 'wb') as file_obj:
            dill.dump(obj , file_obj)
    
    except Exception as e:
        raise CustomException(e , sys)
    
def evaluate_models(X_train, X_test , y_train ,y_test , models , param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train) # Training the Model

            y_train_pred = model.predict(X_train)  # Prediction
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train , y_train_pred)   #Model Evaluation Score
            test_model_score = r2_score(y_test , y_test_pred)

            report[list(models.keys())[i]] = test_model_score

            return report 
    except Exception as e:
      raise CustomException(e,sys)


def load_object(file_path):
    try:
        with open(file_path , 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
        
    