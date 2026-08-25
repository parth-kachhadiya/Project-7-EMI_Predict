import sys
from src.exception import MyException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import EvaluateModel
from src.components.model_pusher import ModelPusher
                                          
from src.entity.artifact_entity import (
    DataIngestionArtifact, 
    DataTransformationArtifact,
    ModelTrainerArtifacts,
    ModelEvaluationArtifacts, 
    ModelPusherArtifacts
)


class TrainingPipeline:


    def s1_do_dataIngestion(self) -> DataIngestionArtifact:
        logging.info("--------------------------------< Ingestion Started >--------------------------------")

        ingestor = DataIngestion()
        ingestion_artifact = ingestor.ingestor()

        logging.info("--------------------------------< Ingestion Done :) >--------------------------------")

        return ingestion_artifact

    def s2_do_dataTransformation(self, ingestionArtifact : DataIngestionArtifact) -> DataTransformationArtifact:
        transformer = DataTransformation(ingestionArtifact)
        transformationArtifact = transformer.transformer()
        return transformationArtifact

    def s3_model_training(self, tArtifact : DataTransformationArtifact) -> ModelTrainerArtifacts:
        trainer = ModelTrainer(tArtifact) 
        artifacts = trainer.trainer()
        return artifacts

    def s4_model_evaluation(self, mtArtifacts : ModelTrainerArtifacts) -> ModelEvaluationArtifacts:
        evaluation = EvaluateModel(mtArtifacts)
        artifacts = evaluation.evaluator()
        return artifacts

    def s5_model_pusher(self, meArtifacts : ModelEvaluationArtifacts) -> ModelPusherArtifacts:
        pusher = ModelPusher(meArtifacts)
        artifacts = pusher.pusher()
        return artifacts

    def run_pipeline(self) -> None:
        try:
            s1_artifact = self.s1_do_dataIngestion()
            s2_artifact = self.s2_do_dataTransformation(s1_artifact)
            s3_artifact = self.s3_model_training(s2_artifact)
            s4_artifact = self.s4_model_evaluation(s3_artifact)
            s5_artifact = self.s5_model_pusher(s4_artifact)
        except Exception as e:
            raise MyException(e, sys)