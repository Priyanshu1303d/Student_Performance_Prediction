import os 
import sys

import pandas as pd
import numpy as np
import dill

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
