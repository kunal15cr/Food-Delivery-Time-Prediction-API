import mlflow
import dagshub
import json
import sys
from mlflow import MlflowClient

# -----------------------------
# Initialize DAGsHub + MLflow
# -----------------------------
dagshub.init(
    repo_owner="kunal15cr",
    repo_name="Food-Delivery-Time-Prediction-API",
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/kunal15cr/Food-Delivery-Time-Prediction-API.mlflow"
)

client = MlflowClient()

# -----------------------------
# Load run information
# -----------------------------
def load_model_information(file_path: str) -> dict:
    with open(file_path) as f:
        return json.load(f)

model_info = load_model_information("run_information.json")
model_name = model_info["model_name"]

SOURCE_STAGE = "Staging"
TARGET_STAGE = "Production"

print(f"🔍 Looking for model '{model_name}' in stage '{SOURCE_STAGE}'")

# -----------------------------
# Get latest model in Staging
# -----------------------------
latest_versions = client.get_latest_versions(
    name=model_name,
    stages=[SOURCE_STAGE]
)

if not latest_versions:
    print(f"❌ No model found in stage '{SOURCE_STAGE}'")
    print("➡️  Make sure a model version is promoted to Staging before this step.")
    sys.exit(1)

latest_version = latest_versions[0].version

print(
    f"🚀 Promoting model '{model_name}' "
    f"version {latest_version} "
    f"from '{SOURCE_STAGE}' to '{TARGET_STAGE}'"
)

# -----------------------------
# Promote model
# -----------------------------
client.transition_model_version_stage(
    name=model_name,
    version=latest_version,
    stage=TARGET_STAGE,
    archive_existing_versions=True
)

print("✅ Model promotion successful!")
