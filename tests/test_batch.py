import numpy as np
import pandas as pd
from wine_project.predict import predict_file

def test_batch_reads_csv_and_writes_predictions(sample_data, model_path, fitted_pipeline, tmp_path):
    x, _ = sample_data
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "result" / "predictions.csv"
    x.iloc[:5].to_csv(input_path, index=False)
    predict_file(model_path, input_path, output_path)
    result = pd.read_csv(output_path)
    assert len(result) == 5
    assert result.columns.tolist() == [*x.columns, "prediction"]
    np.testing.assert_array_equal(result["prediction"], fitted_pipeline.predict(x.iloc[:5]))
