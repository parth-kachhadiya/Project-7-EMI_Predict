from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    trained_file_path:str 
    test_file_path:str
    train_clf:str
    test_clf:str
    train_reg:str
    test_reg:str