import numpy as np
from src.utils.main_utils import evaluate_clf, evaluate_reg, critical_error_rate


def test_evaluate_reg_returns_expected_keys():
    y_true = [100, 200, 300, 400]
    y_pred = [110, 190, 290, 410]

    result = evaluate_reg(y_true, y_pred)

    assert "rmse" in result
    assert "mae" in result
    assert "r2" in result
    assert "mape" in result
    assert isinstance(result["rmse"], float)


def test_evaluate_clf_returns_expected_keys():
    labels = ["Not_Eligible", "High_Risk", "Eligible"]
    y_true = ["Eligible", "High_Risk", "Not_Eligible", "Eligible"]
    y_pred = ["Eligible", "Not_Eligible", "Not_Eligible", "Eligible"]

    result = evaluate_clf(y_true, y_pred, labels)

    assert "accuracy" in result
    assert "f1_macro" in result
    assert "critical_error_rate" in result
    assert isinstance(result["accuracy"], float)


def test_critical_error_rate_no_high_risk_cases():
    labels = ["Not_Eligible", "High_Risk", "Eligible"]
    y_true = ["Eligible", "Not_Eligible"]
    y_pred = ["Eligible", "Not_Eligible"]

    rate, cm = critical_error_rate(y_true, y_pred, labels)

    assert rate == 0