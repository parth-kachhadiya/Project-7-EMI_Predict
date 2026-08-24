import pandas as pd
from abc import ABC, abstractmethod
from src.exception import MyException
import sys


class DataLoader(ABC):

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        raise NotImplementedError


class CSVLoader(DataLoader):

    def __init__(self, source_path : str):
        self.file_path = source_path

    def load_data(self):
        try:
            return pd.read_csv(self.file_path)
        except Exception as e:
            raise MyException(e, sys)

            