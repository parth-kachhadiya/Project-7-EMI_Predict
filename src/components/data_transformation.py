import numpy as np
import pandas as pd
import sys
import os
from typing import Annotated
import joblib

from src.constants import (DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR)
from src.exception import MyException
from src.logger import logging
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact

from src.constants import (
    CONTINUES_NUMERICAL_COLS,
    DISCREATE_NUMERICAL_COLS,
    BINARY_TEXT_COLS,
    MULTICLASS_TEXT_COLS
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, LabelEncoder

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


class DataTransformation:

    def __init__(self, artifact : DataIngestionArtifact, configs : DataTransformationConfig = DataTransformationConfig()):
        try:
            self.iArtifact = artifact
            self._config = configs
        except Exception as e:
            raise MyException(e, sys)

    def _read_data(self, path : str) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except Exception as e:
            raise MyException(e, sys)
    
    def _get_transformer(self) -> ColumnTransformer:
        try:
            continues_numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ])

            discrete_numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ])

            multiclass_text_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ])

            binary_text_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(drop='if_binary', handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer(transformers=[
                ('continues_num', continues_numerical_pipeline, CONTINUES_NUMERICAL_COLS),
                ('discrete_num', discrete_numerical_pipeline, DISCREATE_NUMERICAL_COLS),
                ('multiclass_text', multiclass_text_pipeline, MULTICLASS_TEXT_COLS),
                ('binary_text', binary_text_pipeline, BINARY_TEXT_COLS)
            ], remainder="passthrough")

            return preprocessor
        except Exception as e:
            raise MyException(e, sys)


    def _resampler(self, x_train_clf : pd.DataFrame, y_train_clf : pd.DataFrame) -> tuple[
        Annotated[pd.DataFrame, "X_train_resampled"],
        Annotated[pd.DataFrame, "y_train_resampled"]
    ]:
        try:
            sampling_strategy_over = {'High_Risk': 35000}  # only oversample HR, leave Eligible at original (~59555)
            sampling_strategy_under = {'Not_Eligible': 100000}

            resample_pipeline = ImbPipeline([
                ('over', SMOTE(sampling_strategy=sampling_strategy_over, random_state=42)),
                ('under', RandomUnderSampler(sampling_strategy=sampling_strategy_under, random_state=42))
            ])

            X_train_resampled, y_train_resampled = resample_pipeline.fit_resample(x_train_clf, y_train_clf)

            return X_train_resampled, y_train_resampled
        except Exception as e:
            raise MyException(e, sys)


    def s1_load_via_artifacts(self) -> tuple[
        Annotated[pd.DataFrame, "x_train_reg"],
        Annotated[pd.DataFrame, "x_test"],
        Annotated[pd.DataFrame, "y_train_clf"],
        Annotated[pd.DataFrame, "y_test_clf"],
        Annotated[pd.DataFrame, "y_train_reg"],
        Annotated[pd.DataFrame, "y_test_reg"]
    ]:
        try:
            logging.info("Started data loading.")
            x_train = self._read_data(self.iArtifact.trained_file_path)
            x_test = self._read_data(self.iArtifact.test_file_path)
            y_train_clf = self._read_data(self.iArtifact.train_clf).squeeze()
            y_test_clf = self._read_data(self.iArtifact.test_clf).squeeze()
            y_train_reg = self._read_data(self.iArtifact.train_reg).squeeze()
            y_test_reg = self._read_data(self.iArtifact.test_reg).squeeze()
            logging.info("All data loaded.")

            return x_train, x_test, y_train_clf, y_test_clf, y_train_reg, y_test_reg 
        except Exception as e:
            raise MyException(e, sys)


    def s2_initialize_all_transformations(
        self,
        x_train     : pd.DataFrame,
        x_test      : pd.DataFrame,
        y_train_clf : pd.DataFrame,
        y_test_clf  : pd.DataFrame
    ):
        try:
            Transformer = self._get_transformer()

            # ------ fit ------
            logging.info("ColumnTransformer started it's work..")
            Transformer.fit(x_train)
            logging.info("ColumnTransformer trained.")

            # ------ transform ------
            x_train_t = Transformer.transform(x_train)
            logging.info("Training data transformed.")
            x_test_t = Transformer.transform(x_test)
            logging.info("Test data transformed.")

            # ------ sampling ------
            logging.info("Data sampling started.")
            x_train_resampled, y_train_clf_resampled = self._resampler(x_train_t.copy(), y_train_clf)
            logging.info("Data sampling done.")

            # ------ label encoding ------
            le = LabelEncoder()
            logging.info("Label encoding started.")
            le.fit(y_train_clf_resampled)
            y_train_cls_final = le.transform(y_train_clf_resampled)
            y_test_cls_final = le.transform(y_test_clf)
            logging.info("Label encoding done.")

            return x_train_t, x_test_t, x_train_resampled, y_train_cls_final, y_test_cls_final, Transformer, le
        except Exception as e:
            raise MyException(e, sys)

    def s3_store_all_objects(
        self,
        x_train_t,
        x_test_t,
        x_train_resampled,
        y_train_cls_final,
        y_test_cls_final,
        y_train_reg,
        y_test_reg,
        transformer,
        label_encoder
    ):
        os.makedirs(os.path.join(self._config.data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR), exist_ok=True)
        os.makedirs(os.path.join(self._config.data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR), exist_ok=True)

        logging.info("Saving all numpy array objects.")
        np.save(self._config.xtrain_t_path, x_train_t)
        np.save(self._config.xtest_t_path, x_test_t)
        np.save(self._config.xtrain_resampled_path, x_train_resampled)
        np.save(self._config.ytrain_cls_final_path, y_train_cls_final)
        np.save(self._config.ytest_cls_final_path, y_test_cls_final)
        np.save(self._config.ytrain_reg, y_train_reg)
        np.save(self._config.ytest_reg, y_test_reg)
        logging.info("All numpy array objects are saved.")

        logging.info("Saving all transformation objects.")
        joblib.dump(transformer, self._config.transformer_object_file_path)
        joblib.dump(label_encoder, self._config.labelencoder_file_path)
        logging.info("All transformation objects are saved.")


    def transformer(self) -> DataTransformationArtifact:
        try:
            logging.info("--------------------------------< Data Transformation Started >--------------------------------")
            x_train, x_test, y_train_clf, y_test_clf, y_train_reg, y_test_reg = self.s1_load_via_artifacts()
            x_train_t, x_test_t, x_train_resampled, y_train_cls_final, y_test_cls_final, Transformer, le = self.s2_initialize_all_transformations(x_train, x_test, y_train_clf, y_test_clf)
            self.s3_store_all_objects(x_train_t, x_test_t, x_train_resampled, y_train_cls_final, y_test_cls_final, y_train_reg, y_test_reg, Transformer, le)
            
            transformationArtifacts = DataTransformationArtifact(
                xtrain_t_path = self._config.xtrain_t_path,
                xtest_t_path = self._config.xtest_t_path,
                xtrain_resampled_path = self._config.xtrain_resampled_path,
                ytrain_cls_final_path = self._config.ytrain_cls_final_path,
                ytest_cls_final_path = self._config.ytest_cls_final_path,
                ytrain_reg = self._config.ytrain_reg,
                ytest_reg = self._config.ytest_reg,
                transformer_object_file_path = self._config.transformer_object_file_path,
                labelencoder_file_path = self._config.labelencoder_file_path
            )
            logging.info("--------------------------------< Data Transformation Done :) >--------------------------------")
            return transformationArtifacts
        except Exception as e:
            raise MyException(e, sys)