# Student Performance Prediction Project 📚
This project implements a machine learning solution to predict student academic performance based on various socio-demographic and academic factors. Using features like parental education, test preparation, and previous scores, the model predicts student performance in mathematics, helping educators identify students who might need additional support.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Dataset Information](#dataset-information)
5. [Model Training and Evaluation](#model-training-and-evaluation)
6. [Contributors](#contributors)
---

## Project Overview
This project utilizes several machine learning algorithms, including:
- **Random Forest**
- **XGBoost**
- **CatBoost**
- **Linear Regression**
- **Support Vector Regression**
- **Decision Trees**

The models are optimized using hyperparameter tuning and evaluated based on R² score and Mean Squared Error.

## Installation
To set up the project on your local machine:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/student-performance-prediction.git
   cd student-performance-prediction
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the student performance dataset and place it in the root directory.

## Usage
1. Run the data ingestion script:
   ```bash
   python src/components/data_ingestion.py
   ```

2. Transform the data:
   ```bash
   python src/components/data_transformation.py
   ```

3. Train the model:
   ```bash
   python src/components/model_trainer.py
   ```

4. Deploy using Docker:
   ```bash
   docker build -t student-performance-prediction .
   docker run -p 5000:5000 student-performance-prediction
   ```

## Dataset Information
The dataset includes various features about students:
1. Gender
2. Race/Ethnicity
3. Parental Level of Education
4. Lunch Type
5. Test Preparation Course
6. Reading and Writing Scores
7. Math Score (Target Variable)

## Model Training and Evaluation
The model pipeline includes:
1. Data preprocessing with encoding of categorical features
2. Model training with cross-validation
3. Hyperparameter optimization using GridSearchCV
4. Performance evaluation using R² score metrics
5. Model deployment using Flask API and Docker

## Contributors
- [Priyanshu Kumar Singh](https://github.com/Priyanshu1303d)
