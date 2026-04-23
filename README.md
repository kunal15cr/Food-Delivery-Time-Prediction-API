# 🚀 Food Delivery Time Prediction API

### *Production-ready ML API to predict delivery ETA and improve on-time performance for food delivery operations.*

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-API-success)
![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-0194E2)
![DVC](https://img.shields.io/badge/DVC-Data_&_Pipeline-8A2BE2)

---

## 📌 Problem Statement

Food delivery platforms must estimate delivery time accurately to:

* set realistic customer expectations,
* optimize rider allocation,
* reduce SLA breaches and cancellations.

Traditional rule-based ETA systems struggle with dynamic conditions (traffic, weather, delivery density, order complexity), causing poor customer experience and operational inefficiency.

---

## 💡 Solution Overview

This project builds an **end-to-end machine learning system** that predicts food delivery time (in minutes) from order and logistics features, and serves predictions through a **FastAPI** endpoint.

### Business Impact

* 📉 Reduces ETA prediction error and improves trust in delivery promises.
* ⚡ Enables faster dispatch decisions through real-time inference.
* 🧠 Supports data-driven operations with reproducible ML workflows.

---

## 🏗️ Architecture

```text
             +-----------------------+
             |  Raw Order Data       |
             | (Swiggy-style schema) |
             +-----------+-----------+
                         |
                         v
              +----------+-----------+
              | Data Cleaning &      |
              | Feature Engineering  |
              +----------+-----------+
                         |
                         v
              +----------+-----------+
              | Trained Regressor +  |
              | Preprocessor (MLflow)|
              +----------+-----------+
                         |
                         v
              +----------+-----------+
              | FastAPI /predict      |
              | JSON -> ETA minutes   |
              +-----------------------+
```

---

## ⚙️ Tech Stack

* **Language:** Python
* **Modeling:** Scikit-learn pipelines, Random Forest / LightGBM
* **Serving:** FastAPI + Uvicorn
* **Experiment Tracking:** MLflow
* **Data Versioning:** DVC
* **Testing & Packaging:** pytest, tox, Makefile, Docker
* **Deployment Target:** AWS

---

## 📊 Data & Insights

* **Dataset:** Swiggy-like delivery dataset (`swiggy.csv`)
* **Target:** `time_taken`

### Feature Groups

* Delivery partner details
* Geospatial coordinates
* Time-based features
* Traffic, weather, and order context

---

## 🤖 Model & Performance

* **Type:** Regression Model
* **Pipeline:** Preprocessing + Model
* **Tracking:** MLflow Model Registry
* **Quality Rule:** MAE ≤ 5 minutes

---

## 🔌 API Usage

### Run API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Health Check

```bash
curl http://127.0.0.1:8000/
```

### Prediction Endpoint

**POST /predict**

#### Sample Request

```json
{
  "ID": "0x4607",
  "Delivery_person_ID": "BANGRES18DEL02",
  "Delivery_person_Age": "29",
  "Delivery_person_Ratings": "4.6",
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
  "multiple_deliveries": "0",
  "Festival": "No",
  "City": "Urban"
}
```

#### cURL

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d @sample_payload.json
```

#### Response

```json
35.42
```

---

## 🚀 Deployment

### Docker

```bash
docker build -t food-delivery-eta-api .
docker run -p 8000:8000 food-delivery-eta-api
```

---

## 🔁 MLOps Pipeline

```
DVC → Train → MLflow → Register Model → Deploy → FastAPI
```

---

## 📌 Key Learnings

* Data preprocessing is critical
* Pipeline prevents training-serving mismatch
* Model registry improves lifecycle management
* Testing ensures production reliability

---

## 🔮 Future Improvements

* Drift detection
* SHAP explainability
* Batch predictions
* CI/CD automation
* GenAI insights layer

---

## 🤝 Contributing

Pull requests and issues are welcome.

---

## 📄 License

See `LICENSE` file.
