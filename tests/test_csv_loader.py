import pandas as pd
import pytest
from src.data_access.csv_loader import CSVLoader


def test_csv_loader_reads_valid_file(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("age,salary\n30,50000\n45,80000\n")

    loader = CSVLoader(str(csv_path))
    df = loader.load_data()

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["age", "salary"]
    assert len(df) == 2


def test_csv_loader_missing_file_raises():
    loader = CSVLoader("nonexistent_file.csv")
    with pytest.raises(Exception):
        loader.load_data()