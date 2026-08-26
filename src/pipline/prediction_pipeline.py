import os
import sys
import joblib

import numpy as np
import pandas as pd

import mlflow
from mlflow.tracking import MlflowClient

from src.exception import MyException
from src.logger import logging
from src.configuration.mlflow_connection import MLFlowConection
from src.entity.config_entity import PredictionPipelineConfig
from src.constants import CLF_REGISTERED_MODEL, REG_REGISTERED_MODEL


class PredictionPipeline:

    def __init__(self, configs : PredictionPipelineConfig = PredictionPipelineConfig()):
        try:
            self._config = configs
            self.connection = MLFlowConection(
                repo_owner = self._config.dagshub_repo_owner,
                repo_name = self._config.dagshub_repo_name
            )
            self.client : MlflowClient = self.connection.get_client()

            self.cache_dir = "artifact/prediction_cache"

            self.clf_model = None
            self.reg_model = None
            self.preprocessor = None
            self.label_encoder = None

            self._download_resources()
            self._load_artifacts()
        except Exception as e:
            raise MyException(e, sys)

    def _download_resources(self):
        try:
            os.makedirs(self.cache_dir, exist_ok=True)

            clf_version = self.client.get_latest_versions(CLF_REGISTERED_MODEL)[0]
            reg_version = self.client.get_latest_versions(REG_REGISTERED_MODEL)[0]

            logging.info("Downloading classification model.")
            mlflow.artifacts.download_artifacts(
                artifact_uri=f"models:/{CLF_REGISTERED_MODEL}/{clf_version.version}",
                dst_path=os.path.join(self.cache_dir, "clf_model")
            )

            logging.info("Downloading regression model.")
            mlflow.artifacts.download_artifacts(
                artifact_uri=f"models:/{REG_REGISTERED_MODEL}/{reg_version.version}",
                dst_path=os.path.join(self.cache_dir, "reg_model")
            )

            logging.info("Downloading preprocessor and label encoder.")
            self.client.download_artifacts(
                clf_version.run_id, "preprocessing", dst_path=self.cache_dir
            )

            logging.info(f"All resources downloaded to {self.cache_dir}")
        except Exception as e:
            raise MyException(e, sys)


    def _load_artifacts(self):
        try:

            self.clf_model = mlflow.pyfunc.load_model(os.path.join(self.cache_dir, "clf_model"))
            self.reg_model = mlflow.pyfunc.load_model(os.path.join(self.cache_dir, "reg_model"))

            preprocessing_dir = os.path.join(self.cache_dir, "preprocessing")
            self.preprocessor = joblib.load(os.path.join(preprocessing_dir, "column_transformer.pkl"))
            self.label_encoder = joblib.load(os.path.join(preprocessing_dir, "label_encoder.pkl"))

            logging.info("All resources loaded to memory.")
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, input_data : dict) -> dict:
        try:
            input_df = pd.DataFrame([input_data])
            transformed = self.preprocessor.transform(input_df)

            clf_pred_encoded = self.clf_model.predict(transformed)
            clf_prediction = self.label_encoder.inverse_transform(clf_pred_encoded)[0]

            reg_prediction = self.reg_model.predict(transformed)

            return {
                "emi_eligibility" : clf_prediction,
                "max_monthly_emi" : float(reg_prediction[0])
            }
        except Exception as e:
            raise MyException(e, sys)