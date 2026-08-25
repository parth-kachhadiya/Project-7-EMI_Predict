import os
import sys
import joblib
import mlflow
import yaml

from src.exception import MyException
from src.logger import logging
from src.configuration.mlflow_connection import MLFlowConection
from src.entity.artifact_entity import ModelEvaluationArtifacts, ModelPusherArtifacts
from src.entity.config_entity import ModelPusherConfig
from src.constants import CLF_REGISTERED_MODEL, REG_REGISTERED_MODEL, CLS_DIR, REG_DIR


class ModelPusher:

    def __init__(self, artifact : ModelEvaluationArtifacts, config : ModelPusherConfig = ModelPusherConfig()):
        try:
            self.artifact = artifact
            self._config = config
            self.connection = MLFlowConection(
                repo_owner = self._config.dagshub_repo_owner,
                repo_name = self._config.dagshub_repo_name
            )
        except Exception as e:
            return MyException(e, sys)

    def _load_matrics(self, destination : str):
        try:
            with open(destination, "r") as f:
                content = yaml.safe_load(f)

            return content
        except Exception as e:
            raise MyException(e, sys)

    def _is_xgb(self, path) -> bool:
        try:
            with open(path, 'r') as f:
                content = yaml.safe_load(f)

            return True if 'xgb' in content['model_name'] else False
        except Exception as e:
            raise MyException(e, sys)

    def _push_model(self, model_path: str, run_name: str, registered_model_name: str, flavor: str, metrix, experiment_name : str) -> str:
        try:
            model = joblib.load(model_path)

            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name = run_name):
                try:
                    mlflow.log_params(model.get_params())
                except Exception as param_err:
                    logging.info(f"Could not log params (non-fatal): {param_err}")
                mlflow.log_metrics(metrix)
                if flavor == "xgboost":
                    mlflow.xgboost.log_model(
                        model, artifact_path="model",
                        registered_model_name=registered_model_name
                    )
                else:
                    mlflow.sklearn.log_model(
                        model, artifact_path="model",
                        registered_model_name=registered_model_name,
                        serialization_format="cloudpickle"
                    )
                run_id = mlflow.active_run().info.run_id

            logging.info(f"Model pushed and registered under '{registered_model_name}' (run_id={run_id})")
            return run_id
        except Exception as e:
            raise MyException(e, sys)

    def pusher(self) -> ModelPusherArtifacts:
        try:
            logging.info("--------------------------------< Model Pusher Started >--------------------------------")

            cls_pusher = False
            reg_pusher = False
            cls_run_id = None
            reg_run_id = None

            if self.artifact.should_push_clf:
                
                logging.info("Classification model beats champion. Pushing to registry.")

                fPath = os.path.join(self.artifact.model_artifact_dir_name, CLS_DIR, "metric.yaml")
                modelMatrix = {k: v for k, v in self._load_matrics(fPath).items() if isinstance(v, (int, float))}
                isXGB = self._is_xgb(fPath)
                
                cls_run_id = self._push_model(
                    self.artifact.clf_path,
                    run_name = "pipeline_classification_challenger",
                    registered_model_name = CLF_REGISTERED_MODEL,
                    flavor = "xgboost" if isXGB else "sklearn",
                    metrix = modelMatrix,
                    experiment_name = "classification experiment"
                )
                cls_pusher = True
            else:
                logging.info("Classification model did not beat champion. Skipping push.")

            if self.artifact.should_push_reg:

                logging.info("Regression model beats champion. Pushing to registry.")

                fPath = os.path.join(self.artifact.model_artifact_dir_name, REG_DIR, "metric.yaml")
                modelMatrix = {k: v for k, v in self._load_matrics(fPath).items() if isinstance(v, (int, float))}
                isXGB = self._is_xgb(fPath)

                reg_run_id = self._push_model(
                    self.artifact.reg_path,
                    run_name = "pipeline_regression_challenger",
                    registered_model_name = REG_REGISTERED_MODEL,
                    flavor= "xgboost" if isXGB else "sklearn",
                    metrix = modelMatrix,
                    experiment_name = "regression experiment"
                )
                reg_pusher = True
            else:
                logging.info("Regression model did not beat champion. Skipping push.")

            pusherArtifacts = ModelPusherArtifacts(
                does_clf_pushed = cls_pusher,
                does_reg_pushed = reg_pusher,
                clf_id = cls_run_id,
                reg_id = reg_run_id
            )

            logging.info("--------------------------------< Model Pusher Done >--------------------------------")

            return pusherArtifacts
        except Exception as e:
            raise MyException(e, sys)