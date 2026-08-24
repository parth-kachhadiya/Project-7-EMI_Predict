import os
from datetime import date



PIPELINE_NAME: str = ""
ARTIFACT_DIR: str = "artifact"

MODEL_FILE_NAME = "model.pkl"

CURRENT_YEAR = date.today().year
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"

CLF_OUTPUT_CLASS_LABELS = ['Not_Eligible', 'High_Risk', 'Eligible']

FILE_NAME: str = "data.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
Y_TRAIN_CLF = "y_train_clf.csv"
Y_TEST_CLF = "y_test_clf.csv"
Y_TRAIN_REG = "y_train_reg.csv"
Y_TEST_REG = "y_test_reg.csv"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")


AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "us-east-1"


"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_SOURCE_FILE_PATH = "data"
DATA_INGESTION_FILE_NAME: str = "emi.csv"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.25


"""
Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

CONTINUES_NUMERICAL_COLS = ['age', 'monthly_salary', 'years_of_employment', 'monthly_rent', 'school_fees', 'college_fees',
                               'travel_expenses', 'groceries_utilities', 'other_monthly_expenses', 'current_emi_amount', 'credit_score', 
                               'bank_balance', 'emergency_fund', 'requested_amount', 'requested_tenure']
DISCREATE_NUMERICAL_COLS = ['family_size', 'dependents']
MULTICLASS_TEXT_COLS = [ 'education', 'employment_type', 'company_type', 'house_type', 'emi_scenario']
BINARY_TEXT_COLS = ['gender', 'marital_status', 'existing_loans']

XTRAIN_T_FNAME = "x_train_t.npy"
XTEST_T_FNAME = "x_test_t.npy"
XTRAIN_RS_FNAME = "x_train_resampled.npy"
YTRAIN_CLSF_FNAME = "y_train_cls_final.npy"
YTEST_CLSF_FNAME = "y_test_cls_final.npy"
YTRAIN_REG_FNAME = "y_train_reg.npy"
YTEST_REG_FNAME = "y_test_reg.npy"

COL_TRANSFORMER_FNAME = "column_transformer.pkl"
LABEL_ENCODER_FNAME = "label_encoder.pkl"

"""
MODEL TRAINER related constant start with MODEL_TRAINER var name
"""
PARAMS_PATH = "model-params.yaml"
MODEL_DIR_PATH = "trained_models"
CLS_DIR = "classification"
REG_DIR = "regression"

"""
MODEL Evaluation related constants
"""
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02   # Minimum performance improvement as compare to previous model.
MODEL_BUCKET_NAME = "my-first-mlops-bucket"
MODEL_PUSHER_S3_KEY = "model-registry"


APP_HOST = "0.0.0.0"
APP_PORT = 5000