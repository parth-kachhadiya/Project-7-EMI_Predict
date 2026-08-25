import os
import sys
from typing import Annotated
import numpy as np
import yaml
import joblib

from concurrent.futures import ThreadPoolExecutor

from src.constants import CLF_OUTPUT_CLASS_LABELS
from src.exception import MyException
from src.logger import logging
from src.entity.config_entity import ModelTrainingConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact, 
    ModelTrainerArtifacts
)
from src.utils.main_utils import (
    evaluate_clf, evaluate_reg
)

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor



class ModelTrainer:

    def __init__(self, tArtifact : DataTransformationArtifact, configs : ModelTrainingConfig = ModelTrainingConfig()):
        self.artifacts = tArtifact
        self._config = configs


    def _load_model_data(self, path : str) -> np.ndarray:
        try:
            data = np.load(path)
            return data
        except Exception as e:
            raise MyException(e, sys)


    def _load_params(self, path : str ):
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            return data
        except Exception as e:
            raise MyException(e, sys)

    def _save_models_artifacts(self, clf_model_details : dict, reg_model_details : dict):
        try:
            os.makedirs(self._config.cls_dir, exist_ok=True)
            os.makedirs(self._config.reg_dir, exist_ok=True)

            clf_model_path = os.path.join(self._config.cls_dir, "model.pkl")
            clf_metrics_path = os.path.join(self._config.cls_dir, "metric.yaml")


            joblib.dump(clf_model_details['clf_model'], clf_model_path)
            clf_report = {
                'model_name': clf_model_details['clf_model_name'],
                **clf_model_details['clf_metrics']
            }
            with open(clf_metrics_path, 'w') as f:
                yaml.safe_dump(clf_report, f)
            logging.info(f"Classification model saved to {clf_model_path}")

            reg_model_path = os.path.join(self._config.reg_dir, "model.pkl")
            reg_metrics_path = os.path.join(self._config.reg_dir, "metric.yaml")
            
            joblib.dump(reg_model_details['reg_model'], reg_model_path)
            reg_report = {
                'model_name': reg_model_details['reg_model_name'],
                **reg_model_details['reg_metrics']
            }
            with open(reg_metrics_path, 'w') as f:
                yaml.safe_dump(reg_report, f)
            logging.info(f"Regression model saved to {reg_model_path}")

        except Exception as e:
            raise MyException(e, sys)

    def train_regressors(self, x_train : np.ndarray, y_train : np.ndarray, x_test : np.ndarray, y_test : np.ndarray):
        try:
            params = self._load_params(self._config.model_params_config_file)['regression-models']
            results = {}

            lr = LinearRegression(**params['linear-regressor'])
            logging.info("Linear regressor - training started")
            lr.fit(x_train, y_train)
            results['linear-regressor'] = {'model': lr, 'metrics': evaluate_reg(y_test, lr.predict(x_test))}
            logging.info("Linear regressor - training done.")
            
            rfr = RandomForestRegressor(**params['random-forest-regressor'])
            logging.info("Random forest regressor - training strated")
            rfr.fit(x_train, y_train)
            results['random-forest-regressor'] = {'model': rfr, 'metrics': evaluate_reg(y_test, rfr.predict(x_test))}
            logging.info("Random forest regressor - training done.")

            xgbr = XGBRegressor(**params['xgb-regressor'])
            logging.info("Xgboost regressor - training started")
            xgbr.fit(x_train, y_train)
            results['xgb-regressor'] = {'model': xgbr, 'metrics': evaluate_reg(y_test, xgbr.predict(x_test))}
            logging.info("Xgboost regressor - training done")
            
            best_name = min(results, key=lambda k: results[k]['metrics']['rmse'])
            logging.info(f"Regression training done. Best: {best_name} (RMSE={results[best_name]['metrics']['rmse']:.2f})")

            return results[best_name]['model'], best_name, results[best_name]['metrics'], results

        except Exception as e:
            raise MyException(e, sys)


    def train_classificatiers(self, x_train : np.ndarray, y_train : np.ndarray, x_test : np.ndarray, y_test : np.ndarray):
        try:
            params = self._load_params(self._config.model_params_config_file)['classification-models']
            label_encoder = joblib.load(self.artifacts.labelencoder_file_path)
            y_test_decoded = label_encoder.inverse_transform(y_test)
            results = {}

            lr = LogisticRegression(**params['logistic-regression'])
            logging.info("Logistic regressor - training started")
            lr.fit(x_train, y_train)
            pred_lr = label_encoder.inverse_transform(lr.predict(x_test))
            results['logistic-regression'] = {'model': lr, 'metrics': evaluate_clf(y_test_decoded, pred_lr, CLF_OUTPUT_CLASS_LABELS)}
            logging.info("Logistic regressor - training done.")

            rfc = RandomForestClassifier(**params['random-forest-classifier'])
            logging.info("Decision tree classifier - training started")
            rfc.fit(x_train, y_train)
            pred_rfc = label_encoder.inverse_transform(rfc.predict(x_test))
            results['random-forest-classifier'] = {'model': rfc, 'metrics': evaluate_clf(y_test_decoded, pred_rfc, CLF_OUTPUT_CLASS_LABELS)}
            logging.info("Decision tree classifier - training done.")

            xgbc = XGBClassifier(**params['xgb-classifier'])
            logging.info("Xgboost classifier - training started")
            xgbc.fit(x_train, y_train)
            pred_xgbc = label_encoder.inverse_transform(xgbc.predict(x_test))
            results['xgb-classifier'] = {'model': xgbc, 'metrics': evaluate_clf(y_test_decoded, pred_xgbc, CLF_OUTPUT_CLASS_LABELS)}
            logging.info("Xgboost classifier - training done.")

            best_name = max(results, key=lambda k: results[k]['metrics']['f1_macro'])
            logging.info(f"Classification training done. Best: {best_name} (F1={results[best_name]['metrics']['f1_macro']:.4f})")

            return results[best_name]['model'], best_name, results[best_name]['metrics'], results

        except Exception as e:
            raise MyException(e, sys)

    def trainer(self) -> ModelTrainerArtifacts:
        try:
            x_train_reg = self._load_model_data(self.artifacts.xtrain_t_path)
            x_test = self._load_model_data(self.artifacts.xtest_t_path)
            y_train_reg = self._load_model_data(self.artifacts.ytrain_reg)
            y_test_reg = self._load_model_data(self.artifacts.ytest_reg)

            x_train_clf = self._load_model_data(self.artifacts.xtrain_resampled_path)
            y_train_clf = self._load_model_data(self.artifacts.ytrain_cls_final_path)
            y_test_clf = self._load_model_data(self.artifacts.ytest_cls_final_path)

            logging.info("--------------< Parallel model training initialized (Classifier + Regressor) >--------------")

            with ThreadPoolExecutor(max_workers=2) as executor:
                reg_future = executor.submit(self.train_regressors, x_train_reg, y_train_reg, x_test, y_test_reg)
                clf_future = executor.submit(self.train_classificatiers, x_train_clf, y_train_clf, x_test, y_test_clf)

                best_reg_model, best_reg_name, best_reg_metrics, all_reg_results = reg_future.result()
                best_clf_model, best_clf_name, best_clf_metrics, all_clf_results = clf_future.result()

            logging.info(f"Best regressor: {best_reg_name} | Best classifier: {best_clf_name}")

            self._save_models_artifacts(
                {
                    'clf_model_name' : best_clf_name,
                    'clf_model' : best_clf_model,
                    'clf_metrics' : best_clf_metrics
                },
                {
                    'reg_model_name' : best_reg_name,
                    'reg_model' : best_reg_model,
                    'reg_metrics' : best_reg_metrics
                }
            )

            logging.info(f"All traces saved to artifacts.....")

            artifacts = ModelTrainerArtifacts(
                model_artifact_dir_name = self._config.model_destination_path
            )

            return artifacts
        except Exception as e:
            raise MyException(e, sys)