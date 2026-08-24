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