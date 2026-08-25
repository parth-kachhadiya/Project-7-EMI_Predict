from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    trained_file_path:str 
    test_file_path:str
    train_clf:str
    test_clf:str
    train_reg:str
    test_reg:str

@dataclass
class DataTransformationArtifact:
    xtrain_t_path: str 
    xtest_t_path: str 
    xtrain_resampled_path: str
    ytrain_cls_final_path: str
    ytest_cls_final_path: str
    ytrain_reg: str
    ytest_reg: str
    transformer_object_file_path: str
    labelencoder_file_path:str

@dataclass
class ModelTrainerArtifacts:
    model_artifact_dir_name: str

@dataclass
class ModelEvaluationArtifacts:
    model_artifact_dir_name: str
    should_push_clf : bool
    should_push_reg : bool
    clf_path : str
    reg_path : str

@dataclass
class ModelPusherArtifacts:
    does_clf_pushed : bool
    does_reg_pushed : bool
    clf_id : str
    reg_id : str
