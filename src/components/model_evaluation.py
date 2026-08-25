import sys
import os
import yaml

from src.exception import MyException
from src.logger import logging
from src.configuration import mlflow_connection
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifacts, ModelEvaluationArtifacts
from src.configuration.mlflow_connection import MLFlowConection
from src.constants import MIN_IMPROVEMENT_NEEDED_CLF, MIN_IMPROVEMENT_NEEDED_REG, CLS_DIR, REG_DIR, CLF_REGISTERED_MODEL, REG_REGISTERED_MODEL

class EvaluateModel:

    def __init__(self, artifact : ModelTrainerArtifacts, configs : ModelEvaluationConfig = ModelEvaluationConfig()):
        try:
            self.artifact = artifact
            self._config = configs
            self.connection = MLFlowConection(
                repo_owner = self._config.dagshub_repo_owner,
                repo_name = self._config.dagshub_repo_name
            )
            
            self._config.model_artifact_dir_name = self.artifact.model_artifact_dir_name
        except Exception as e:
            raise MyException(e, sys)

    def _load_matrix(self, destination : str):
        try:
            file_path = os.path.join(self.artifact.model_artifact_dir_name, destination, "metric.yaml")

            with open(file_path, "r") as f:
                content = yaml.safe_load(f)

            return content
        except Exception as e:
            raise MyException(e, sys)
    

    def _which_regressor_is_better(self, new_rmse : float, current_champ_score : float) -> bool:
        try:
            if current_champ_score is None:
                logging.info(f"No current chamption found, marking current as banchmark.")
                return True
            relImprovement = (current_champ_score - new_rmse) / current_champ_score
            logging.info(f"Regression RMSE comparision : current chamption = {current_champ_score:.4f}, new = {new_rmse:.4f}, improvement : {relImprovement:.4f}")
            
            return relImprovement > MIN_IMPROVEMENT_NEEDED_REG
        except Exception as e:
            raise MyException(e, sys)

    def _which_classifier_is_better(self, new_f1 : float, current_champ_f1 : float) -> bool:
        try:
            if current_champ_f1 is None:
                logging.info("No existing classification champion found. Marking current as banchmark.")
                return True

            improvement = new_f1 - current_champ_f1
            logging.info(f"Classification F1 comparison -> new: {new_f1:.4f}, champion: {current_champ_f1:.4f}, delta: {improvement:.4f}")

            return improvement > MIN_IMPROVEMENT_NEEDED_CLF
            
        except Exception as e:
            raise MyException(e ,sys)


    def evaluate_classifier(self) -> tuple[bool, float]:
        try:
            new_f1 = self._load_matrix(CLS_DIR)['f1_macro']

            champ_f1 = self.connection.get_chamption_matric(
                CLF_REGISTERED_MODEL, "f1_macro"
            )

            should_push = self._which_classifier_is_better(new_f1, champ_f1)

            return should_push, new_f1
        except Exception as e:
            raise MyException(e, sys)

    def evaluate_regressor(self) -> tuple[bool, float]:
        try:
            new_rmse = self._load_matrix(REG_DIR)['rmse']

            champ_rmse = self.connection.get_chamption_matric(
                REG_REGISTERED_MODEL, "rmse"
            )

            should_push = self._which_regressor_is_better(new_rmse, champ_rmse)

            return should_push, new_rmse
        except Exception as e:
            raise MyException(e, sys)

    def evaluator(self) -> ModelEvaluationArtifacts:
        try:
            logging.info("--------------------------------< Model Evaluation Started >--------------------------------")

            should_push_clf, f1 = self.evaluate_classifier()
            should_push_reg, rmse = self.evaluate_regressor()

            logging.info(f"Classification push decision: {should_push_clf} (F1={f1:.4f})")
            logging.info(f"Regression push decision: {should_push_reg} (RMSE={rmse:.2f})")

            artifacts = ModelEvaluationArtifacts(
                model_artifact_dir_name = self._config.model_artifact_dir_name,
                should_push_clf = should_push_clf,
                should_push_reg = should_push_reg,
                clf_path = os.path.join(self.artifact.model_artifact_dir_name, CLS_DIR, "model.pkl"),
                reg_path = os.path.join(self.artifact.model_artifact_dir_name, REG_DIR, "model.pkl")
            )

            logging.info("--------------------------------< Model Evaluation Done >--------------------------------")
            return artifacts
        except Exception as e:
            raise MyException(e, sys)