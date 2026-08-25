import os
from src.constants import *
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
    training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
    train_clf: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, Y_TRAIN_CLF)
    test_clf: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, Y_TEST_CLF)
    train_reg: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, Y_TRAIN_REG)
    test_reg: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, Y_TEST_REG)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    source_file_loc: str = os.path.join(DATA_INGESTION_SOURCE_FILE_PATH, DATA_INGESTION_FILE_NAME)


@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
    xtrain_t_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, XTRAIN_T_FNAME)
    xtest_t_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, XTEST_T_FNAME)
    xtrain_resampled_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, XTRAIN_RS_FNAME)
    ytrain_cls_final_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, YTRAIN_CLSF_FNAME)
    ytest_cls_final_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, YTEST_CLSF_FNAME)
    ytrain_reg: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, YTRAIN_REG_FNAME)
    ytest_reg: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, YTEST_REG_FNAME)

    transformer_object_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR, COL_TRANSFORMER_FNAME)
    labelencoder_file_path:str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR, LABEL_ENCODER_FNAME)


@dataclass
class ModelTrainingConfig:
    model_params_config_file: str = PARAMS_PATH
    model_destination_path: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_DIR_PATH)
    cls_dir: str = os.path.join(model_destination_path, CLS_DIR)
    reg_dir: str = os.path.join(model_destination_path, REG_DIR)

@dataclass
class ModelEvaluationConfig:
    dagshub_repo_name : str = DAGSHUB_REPO
    dagshub_repo_owner : str = DAGSHUB_OWNER