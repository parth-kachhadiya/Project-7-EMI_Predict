import sys
import dagshub
import mlflow
from mlflow.tracking import MlflowClient
from src.exception import MyException
from src.logger import logging

class MLFlowConection:
    """Handles DagsHub/MLflow connection setup and exposes a reusable client."""

    def __init__(self, repo_owner : str, repo_name : str):
        try:
            self._owner = repo_owner
            self._name = repo_name
            self._client = None
            self._connect()
        except Exception as e:
            raise MyException(e, sys)

    def _connect(self):
        try:
            dagshub.init(repo_owner=self._owner, repo_name=self._name, mlflow=True)
            self._client = MlflowClient()
            logging.info("MLflow/DagsHub connection established.")
        except Exception as e:
            raise MyException(e, sys)

    def get_client(self) -> MlflowClient:
        return self._client

    def get_chamption_matric(self, registered_model_name : str, param_metric_key : str):
        """
        Returns the champion's metric value for the given registered model,
        or None if no version is registered yet (first-run case).
        """
        try:
            versions = self._client.get_latest_versions(registered_model_name)
            if not versions:
                logging.info(f"No existing champion found for '{registered_model_name}'. First push expected.")
                return None

            latest = versions[0]  
            run = self._client.get_run(latest.run_id)
            metric_value = run.data.metrics.get(param_metric_key)
            return metric_value
        except Exception as e:
            raise MyException(e, sys)