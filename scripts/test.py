import dagshub
import mlflow
import mlflow.pyfunc
import pandas as pd
from pathlib import Path

# 1. Init DAGsHub (auth + config)
dagshub.init(
    repo_owner="kunal15cr",
    repo_name="Food-Delivery-Time-Prediction-API",
    mlflow=True
)

# 2. Set MLflow tracking URI
mlflow.set_tracking_uri(
    "https://dagshub.com/kunal15cr/Food-Delivery-Time-Prediction-API.mlflow"
)

root_path =  Path().resolve()

# 3. Load model from Model Registry
MODEL_URI = "models:/delivery_time_pred_model/latest"

model = mlflow.pyfunc.load_model(MODEL_URI)

print("Model loaded from DAGsHub successfully")

test_data_path = root_path / "data" / "processed" / "test_trans.csv"

df = pd.read_csv(test_data_path).head()

df.head()
y = df.drop(columns=["time_taken"])

print(model.predict(y))
print(len(y.columns))
print(y.columns)

