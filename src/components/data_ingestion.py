import os
import sys

import pandas as pd 
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import MyException
from src.logger import logging
from src.data_access.csv_loader import CSVLoader
from src.utils.main_utils import sanitize_numeric_string, sanitize_gender


class DataIngestion:

    def __init__(self, configs : DataIngestionConfig = DataIngestionConfig()):
        try:
            self._config = configs
        except Exception as e:
            raise MyException(e, sys)

    def s1_export_data(self) -> pd.DataFrame:
        try:
            logging.info("Loading data from csv file.")
            csvLoader = CSVLoader(self._config.source_file_loc)
            data = csvLoader.load_data()
            logging.info("Data loaded successfully to RAM.")
            return data
        except Exception as e:
            raise MyException(e, sys)

    def s2_basic_sanitization(self, data : pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Sanitization process started.")
            
            for col in ['age', 'monthly_salary', 'bank_balance']:
                data[col] = data[col].apply(sanitize_numeric_string)
                data[col] = pd.to_numeric(data[col], errors='coerce')
            logging.info("Numerical sanitization done.")

            data['gender'] = data['gender'].apply(sanitize_gender)
            logging.info("Gender sanitization done.")

            return data
        except Exception as e:
            raise MyException(e, sys)

    def s3_data_split_store(self, data : pd.DataFrame) -> None:
        try:
            logging.info("Data splitting initialized.")
            X = data.drop(columns=['emi_eligibility', 'max_monthly_emi'])
            y_clf = data['emi_eligibility']
            y_reg = data['max_monthly_emi']

            x_train, x_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = train_test_split(
                X, y_clf, y_reg,
                test_size = self._config.train_test_split_ratio, 
                random_state = 34, stratify = y_clf
            )
            
            dir_path = os.path.dirname(self._config.training_file_path)
            os.makedirs(dir_path,exist_ok=True)

            logging.info("Data splitted, preparing to save.")
            x_train.to_csv(self._config.training_file_path, index=False, header=True)
            x_test.to_csv(self._config.testing_file_path, index=False, header=True)
            y_clf_train.to_csv(self._config.train_clf, index=False, header=True)
            y_clf_test.to_csv(self._config.test_clf, index=False, header=True)
            y_reg_train.to_csv(self._config.train_reg, index=False, header=True)
            y_reg_test.to_csv(self._config.test_reg, index=False, header=True)
            logging.info(f"Ingested data saved to {self._config.data_ingestion_dir}")

        except Exception as e:
            raise MyException(e, sys)

    def ingestor(self) -> DataIngestionArtifact:
        try:
            sourceData = self.s1_export_data()
            sanitizedData = self.s2_basic_sanitization(sourceData)
            self.s3_data_split_store(sanitizedData)

            ingestionArtifact = DataIngestionArtifact(
                trained_file_path = self._config.training_file_path,
                test_file_path = self._config.testing_file_path,
                train_clf = self._config.train_clf,
                test_clf = self._config.test_clf,
                train_reg = self._config.train_reg,
                test_reg = self._config.test_reg
            )

            return ingestionArtifact
        except Exception as e:
            raise MyException(e, sys)