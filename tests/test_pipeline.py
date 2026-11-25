import numpy as np
import pytest
from wine_project.model import make_pipeline, save_pipeline, load_pipeline

def test_preprocessing_removes_nan_and_preserves_shape(sample_data, fitted_pipeline):
    x, _ = sample_data
    transformed = fitted_pipeline[:-1].transform(x)
    assert transformed.shape == x.shape
    assert np.isfinite(transformed).all()

def test_predict_returns_one_label_per_row(sample_data, fitted_pipeline):
    x, _ = sample_data
    prediction = fitted_pipeline.predict(x.iloc[:4])
    assert prediction.shape == (4,)
    assert set(prediction).issubset({0, 1, 2})

def test_predict_does_not_refit_scaler(sample_data, fitted_pipeline):
    x, _ = sample_data
    before = fitted_pipeline.named_steps["scaler"].mean_.copy()
    fitted_pipeline.predict(x.iloc[:2] * 10)
    np.testing.assert_array_equal(fitted_pipeline.named_steps["scaler"].mean_, before)

def test_pickle_preserves_predictions_and_probabilities(sample_data, fitted_pipeline, tmp_path):
    x, _ = sample_data
    path = tmp_path / "nested" / "model.pkl"
    save_pipeline(fitted_pipeline, path)
    restored = load_pipeline(path)
    np.testing.assert_array_equal(restored.predict(x), fitted_pipeline.predict(x))
    np.testing.assert_allclose(restored.predict_proba(x), fitted_pipeline.predict_proba(x))

@pytest.mark.parametrize("c", [0, -1])
def test_invalid_regularization_is_rejected(c):
    with pytest.raises(ValueError, match="positive"):
        make_pipeline(c)
