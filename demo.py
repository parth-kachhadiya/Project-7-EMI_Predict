# from src.pipline.training_pipeline import TrainingPipeline

# TrainingPipeline().run_pipeline()

from src.components.model_evaluation import EvaluateModel
from src.entity.artifact_entity import ModelTrainerArtifacts

modelTrainerArtifact = ModelTrainerArtifacts(
    "artifact\\08_24_2026_22_05_57\\trained_models"
)

obj = EvaluateModel(modelTrainerArtifact)
artifacts = obj.evaluator()
print(artifacts)