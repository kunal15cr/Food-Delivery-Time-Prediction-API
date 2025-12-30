from mlflow.tracking import MlflowClient
import mlflow

mlflow.set_tracking_uri("https://dagshub.com/kunal15cr/Food-Delivery-Time-Prediction-API.mlflow")
client = MlflowClient()
run_id = "0f633ea5cb374e6ca8cfee750cb752d5"

print(f"Listing artifacts for run {run_id}")

def walk(path=None, indent=0):
    artifacts = client.list_artifacts(run_id, path=path)
    for a in artifacts:
        print(' ' * indent + f"- {a.path} (dir={a.is_dir})")
        if a.is_dir:
            walk(a.path, indent + 2)

walk()
