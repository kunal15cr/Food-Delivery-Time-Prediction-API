import pandas as pd
import numpy as np


# --------------------------------------------------
# Column standardization
# --------------------------------------------------
def change_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names and make them model-friendly
    """
    return (
        data
        .rename(columns=str.lower)
        .rename(columns={
            "delivery_person_id": "rider_id",
            "delivery_person_age": "age",
            "delivery_person_ratings": "ratings",
            "delivery_location_latitude": "delivery_latitude",
            "delivery_location_longitude": "delivery_longitude",
            "time_taken(min)": "time_taken",
        })
    )


# --------------------------------------------------
# Core data cleaning
# --------------------------------------------------
def data_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean invalid rows, cast datatypes, and engineer base features
    """
    return (
        data
        .drop(columns=["id"], errors="ignore")
        .replace(["NaN", "NaN "], np.nan)
        .assign(
            age=lambda x: pd.to_numeric(x["age"], errors="coerce"),
            ratings=lambda x: pd.to_numeric(x["ratings"], errors="coerce"),
            time_taken=lambda x: pd.to_numeric(x["time_taken"], errors="coerce"),
        )
        # Remove minors and invalid ratings
        .loc[lambda x: (x["age"] >= 18) & (x["ratings"] != 6)]
        .assign(
            city_name=lambda x: x["rider_id"].str.split("RES").str[0],
            restaurant_latitude=lambda x: x["restaurant_latitude"].abs(),
            restaurant_longitude=lambda x: x["restaurant_longitude"].abs(),
            delivery_latitude=lambda x: x["delivery_latitude"].abs(),
            delivery_longitude=lambda x: x["delivery_longitude"].abs(),
        )
    )


# --------------------------------------------------
# Latitude / Longitude cleanup
# --------------------------------------------------
def clean_lat_long(data: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    """
    Remove invalid latitude/longitude values
    """
    location_columns = [
        "restaurant_latitude",
        "restaurant_longitude",
        "delivery_latitude",
        "delivery_longitude",
    ]

    return data.assign(**{
        col: lambda x, c=col: np.where(x[c] < threshold, np.nan, x[c])
        for col in location_columns
    })


# --------------------------------------------------
# Distance calculation
# --------------------------------------------------
def calculate_haversine_distance(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate delivery distance using Haversine formula (km)
    """
    lat1, lon1, lat2, lon2 = map(
        np.radians,
        [
            data["restaurant_latitude"],
            data["restaurant_longitude"],
            data["delivery_latitude"],
            data["delivery_longitude"],
        ]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    distance = 6371 * 2 * np.arcsin(np.sqrt(a))

    return data.assign(distance=distance)


# --------------------------------------------------
# Export (PIPE SAFE)
# --------------------------------------------------
def export_data_cleaning_pipeline(data: pd.DataFrame) -> pd.DataFrame:
    """
    Export cleaned data to CSV and return DataFrame
    """
    data.to_csv("Food_Delivery_cleaned_data.csv", index=False)
    return data


def create_distance_types(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create categorical distance types based on distance
    """
    return data.assign(
        distance_type=lambda x: pd.cut(
            x["distance"],
            bins=[0, 5, 10, 15, 25],
            labels=["short", "medium", "long", "very long"]
        )
    )


# --------------------------------------------------
# Full pipeline
# --------------------------------------------------
def full_data_cleaning_pipeline(data: pd.DataFrame) -> pd.DataFrame:
    """
    End-to-end data cleaning pipeline
    """
    return (
        data
        .pipe(change_column_names)
        .pipe(data_cleaning)
        .pipe(clean_lat_long)
        .pipe(calculate_haversine_distance)
        .pipe(create_distance_types)
        .pipe(export_data_cleaning_pipeline)
        
    )

if __name__ == "__main__":
    # Example usage
    raw_data = pd.read_excel("Food Delivery Time Prediction Case Study.xlsx", sheet_name="Sheet1")
    cleaned_data = full_data_cleaning_pipeline(raw_data)
    cleaned_data.to_csv("Food_Delivery_cleaned_data.csv", index=False)
