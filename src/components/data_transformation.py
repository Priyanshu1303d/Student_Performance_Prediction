import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        '''
        This function is responsible for data transformation
        '''


        try:
            numerical_columns = ['writting_score', 'reading_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy="median")),
                    ('scaler', StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy="most_frequent")),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical Columns are : {categorical_columns}")
            logging.info(f"Numerical Columns are:{numerical_columns}")


            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)  #train.csv
            test_df = pd.read_csv(test_path)    #test.csv

            logging.info("Train and test data reading completed successfully")

            # we are getting preprocessor by this funciton
            preprocessor_obj = self.get_data_transformation_object()

            #separating for X and Y columns
            target_columns = 'math_score'
            numerical_columns = ['writting_score', 'reading_score']

            #splitting X and Y for train dataset (train_df)
            input_feature_train_df = train_df.drop(columns = [target_columns], axis = 1)
            target_feature_train_df = train_df[target_columns]

            #splitting X and Y for test dataset (test_df)
            input_feature_test_df = test_df.drop(columns = [target_columns], axis = 1)
            target_feature_test_df = test_df[target_columns]

            logging.info(f"Applying preprocessing object on training and testing dataFrame")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            #combining array horizontally
            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr , np.array(target_feature_test_df)
            ]

            logging.info(f"Saving Preprocessing Object")

            #saving pickle file for preprocessing object using function defined in utils
            save_object (
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )
            return (
                train_arr, 
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e , sys)
            