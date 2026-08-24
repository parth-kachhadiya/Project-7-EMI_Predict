import sys
from src.exception import MyException
from src.logger import logging

from src.components.data_ingestion import DataIngestion

from src.entity.config_entity import (DataIngestionConfig)
                                          
from src.entity.artifact_entity import (DataIngestionArtifact)


class TrainingPipeline:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()


    def s1_do_dataIngestion(self) -> DataIngestionArtifact:
        logging.info("--------------------------------< Ingestion Started >--------------------------------")

        ingestor = DataIngestion(self.ingestion_config)
        ingestion_artifact = ingestor.ingestor()

        logging.info("--------------------------------< Ingestion Done >--------------------------------")

        return ingestion_artifact


    def run_pipeline(self) -> None:
        try:
            s1_artifact = self.s1_do_dataIngestion()
        except Exception as e:
            raise MyException(e, sys)