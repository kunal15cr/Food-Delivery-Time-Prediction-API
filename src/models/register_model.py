import mlflow
import dagshub
import json
from pathlib import Path
from mlflow import MlflowClient
import logging

# -------------------- LOGGER SETUP --------------------
logger = logging.getLogger("register_model")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# -------------------- DAGSHUB + MLFLOW INIT --------------------
dagshub.init(
    repo_owner="kunal15cr",
    repo_name="Food-Delivery-Time-Prediction-API",
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/kunal15cr/Food-Delivery-Time-Prediction-API.mlflow"
)

# -------------------- UTILS --------------------
def load_run_information(file_path: Path) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(f"run_information.json not found at {file_path}")

    with open(file_path, "r") as f:
        return json.load(f)

# -------------------- MAIN --------------------
if __name__ == "__main__":
    # project root
    root_path = Path(__file__).parent.parent.parent

    # run info file
    run_info_path = root_path / "run_information.json"

    run_info = load_run_information(run_info_path)

    run_id = run_info["run_id"]
    artifact_path = run_info["artifact_path"]
    model_name = run_info["model_name"]

    # 🔥 THIS IS THE ONLY CORRECT MODEL URI FORMAT
    model_uri = f"https://dagshub.com/kunal15cr/Food-Delivery-Time-Prediction-API.mlflow/#/experiments/6/runs/{run_id}/artifacts/{artifact_path}"

    logger.info(f"Registering model from URI: {model_uri}")
    logger.info(f"Model registry name: {model_name}")

    # -------------------- REGISTER MODEL --------------------
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )

    logger.info(
        f"Registered model '{model_name}' "
        f"as version {model_version.version}"
    )

    # -------------------- TRANSITION TO STAGING --------------------
    client = MlflowClient()

    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Staging",
        archive_existing_versions=False
    )

    logger.info(
        f"Model '{model_name}' version {model_version.version} "
        f"moved to STAGING"
    )
