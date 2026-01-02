import pytest
import mlflow
import dagshub
import json
from pathlib import Path
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error

dagshub.init( repo_owner="kunal15cr", repo_name="Food-Delivery-Time-Prediction-API", mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/kunal15cr/Food-Delivery-Time-Prediction-API.mlflow"
)



def load_model_from_dagshub(model_name: str, model_version: str):
    model_name=model_name
    model_version= model_version

    model_uri=f"models:/{model_name}/{model_version}"

    model=mlflow.sklearn.load_model(model_uri)
    model
    return model

def load_transformer(transformer_path):
    transformer = joblib.load(transformer_path)
    return transformer


def model_pipline(model, preprocessor):
    pipeline = Pipeline(steps=[
    ('preprocess',preprocessor),
    ("regressor",model)
    ])
    return pipeline

pridict_model = load_model_from_dagshub("delivery_time_pred_model", "latest")

root_path =  Path().resolve()

preprocessor_path = root_path / "models" / "preprocessor.joblib" 

preprocessor = load_transformer(preprocessor_path)

pridict_pipeline = model_pipline(pridict_model, preprocessor)

test_data_path = root_path / "data" / "interim" / "test.csv"

@pytest.mark.parametrize(argnames="model_pipe, test_data_path, threshold_error",
                        argvalues=[(pridict_pipeline, test_data_path, 5)])
def test_model_performance(model_pipe,test_data_path,threshold_error):
    # load test data
    df = pd.read_csv(test_data_path)
    
    # drop the missing values
    df.dropna(inplace=True)
    
    # make X and y
    X = df.drop(columns=["time_taken"])
    y = df['time_taken']
    
    # get the predictions
    y_pred = model_pipe.predict(X)
    
    # calculate the mean error
    mean_error = mean_absolute_error(y,y_pred)
    
    # check for performance
    assert mean_error <= threshold_error, f"The model does not pass the performance threshold of {threshold_error} minutes"
    print("The avg error is", mean_error)
    
    print(f"The delivery_time_pred_model model passed the performance test")
     