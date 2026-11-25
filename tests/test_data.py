import numpy as np
import pandas as pd
import pytest
import wine_project.data as data

def test_preparation_selects_and_orders_features(sample_data):
    x, _ = sample_data
    frame = x[list(reversed(data.FEATURES))].assign(target=0, note="ignore")
    result = data.prepare_features(frame)
    assert result.columns.tolist() == data.FEATURES
    pd.testing.assert_frame_equal(result, x)

def test_preparation_does_not_change_input(sample_data):
    x, _ = sample_data
    original = x.copy(deep=True)
    data.prepare_features(x)
    pd.testing.assert_frame_equal(x, original)

@pytest.mark.parametrize("missing", data.FEATURES)
def test_missing_feature_has_clear_error(sample_data, missing):
    x, _ = sample_data
    with pytest.raises(ValueError, match=missing):
        data.prepare_features(x.drop(columns=missing))

def test_numeric_strings_are_converted(sample_data):
    x, _ = sample_data
    frame = x.copy()
    frame["alcohol"] = frame["alcohol"].astype(str)
    result = data.prepare_features(frame)
    np.testing.assert_allclose(result["alcohol"], x["alcohol"])

def test_invalid_string_is_rejected(sample_data):
    x, _ = sample_data
    x["alcohol"] = "not a number"
    with pytest.raises(ValueError):
        data.prepare_features(x)

@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_infinite_values_are_rejected(sample_data, value):
    x, _ = sample_data
    x.loc[0, "alcohol"] = value
    with pytest.raises(ValueError, match="Infinite"):
        data.prepare_features(x)

