import sys
from src.exception import MyException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.entity.config_entity import (DataIngestionConfig)
                                          
from src.entity.artifact_entity import (DataIngestionArtifact, DataTransformationArtifact, ModelTrainerArtifacts)


class TrainingPipeline:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()


    def s1_do_dataIngestion(self) -> DataIngestionArtifact:
        logging.info("--------------------------------< Ingestion Started >--------------------------------")

        ingestor = DataIngestion(self.ingestion_config)
        ingestion_artifact = ingestor.ingestor()

        logging.info("--------------------------------< Ingestion Done :) >--------------------------------")

        return ingestion_artifact

    def s2_do_dataTransformation(self, ingestionArtifact : DataIngestionArtifact) -> DataTransformationArtifact:
        transformer = DataTransformation(ingestionArtifact)
        transformationArtifact = transformer.transformer()
        return transformationArtifact

    def s3_model_training(self, tArtifact : DataTransformationArtifact) -> ModelTrainerArtifacts:
        trainer = ModelTrainer(tArtifact) 
        trainer.trainer()

    def run_pipeline(self) -> None:
        try:
            s1_artifact = self.s1_do_dataIngestion()
            s2_artifact = self.s2_do_dataTransformation(s1_artifact)
            s3_artifact = self.s3_model_training(s2_artifact)
        except Exception as e:
            raise MyException(e, sys)