import mlflow.sklearn
import mlflow
import dagshub
import json
from fastapi import FastAPI
import uvicorn
from pathlib import Path
from mlflow import MlflowClient
import logging
import joblib
import numpy as np
from typing import List
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from data_clean_utils import perform_data_cleaning




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

app = FastAPI()


# create the predict endpoint

def make_prediction(data: dict[str, float]) -> float:
    X = np.array([[
        data["ID"],
        data["Delivery_person_ID"],
        data["Delivery_person_Age"],
        data["Delivery_person_Ratings"],
        data["Restaurant_latitude"],
        data["Restaurant_longitude"],
        data["Delivery_location_latitude"],
        data["Delivery_location_longitude"],
        data["Order_Date"],
        data["Time_Orderd"],
        data["Time_Order_picked"],
        data["Weatherconditions"],
        data["Road_traffic_density"],
        data["Vehicle_condition"],
        data["Type_of_order"],
        data["Type_of_vehicle"],
        data["multiple_deliveries"],
        data["Festival"],
        data["City"]
    ]])   # IMPORTANT: keep mixed types
    X = pd.DataFrame(
        X,
        columns=[
            "ID",
            "Delivery_person_ID",
            "Delivery_person_Age",
            "Delivery_person_Ratings",
            "restaurant_latitude",
            "restaurant_longitude",
            "delivery_latitude",
            "delivery_longitude",
            "order_date",
            "order_time",
            "order_picked_time",
            "weather",
            "traffic",
            "vehicle_condition",
            "type_of_order",
            "type_of_vehicle",
            "multiple_deliveries",
            "festival",
            "city_type"
        ]
    )   
    cleaned_data = perform_data_cleaning(X)

    return pridict_pipeline.predict(cleaned_data)[0]


test_data = {
    "ID": "0x4607",
    "Delivery_person_ID": "BANGRES18DEL02",
    "Delivery_person_Age": 29,
    "Delivery_person_Ratings": 4.6,
    "Restaurant_latitude": 12.9716,
    "Restaurant_longitude": 77.5946,
    "Delivery_location_latitude": 12.9352,
    "Delivery_location_longitude": 77.6245,
    "Order_Date": "2022-03-15",
    "Time_Orderd": "19:45:00",
    "Time_Order_picked": "19:55:00",
    "Weatherconditions": "Sunny",
    "Road_traffic_density": "High",
    "Vehicle_condition": 8,
    "Type_of_order": "Meal",
    "Type_of_vehicle": "Motorcycle",
    "multiple_deliveries": 0,
    "Festival": "No",
    "City": "Urban"
}

make_prediction(test_data)