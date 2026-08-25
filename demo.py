# from src.pipline.training_pipeline import TrainingPipeline

# TrainingPipeline().run_pipeline()

from src.components.model_evaluation import EvaluateModel
from src.entity.artifact_entity import ModelTrainerArtifacts
from src.components.model_pusher import ModelPusher


modelTrainerArtifact = ModelTrainerArtifacts(
    "artifact\\08_25_2026_20_14_22\\trained_models"
)

obj = EvaluateModel(modelTrainerArtifact)
artifacts = obj.evaluator()

obj2 = ModelPusher(artifacts)
artifacts2 = obj2.pusher()
print(artifacts2)
