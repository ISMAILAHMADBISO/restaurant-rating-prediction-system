
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "rating_predictor.joblib"


def load_model():
    """Load the trained machine learning model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run python train.py first."
        )

    return joblib.load(MODEL_PATH)


def predict_rating(
    restaurant_name,
    cuisines,
    votes,
    average_cost,
    price_range,
    city,
    country_code,
    currency,
    table_booking,
    online_delivery,
    longitude,
    latitude,
):
    """Predict a restaurant's aggregate rating."""

    model = load_model()

    input_data = pd.DataFrame([
        {
            "Votes": votes,
            "Average Cost for two": average_cost,
            "Price range": price_range,
            "Longitude": longitude,
            "Latitude": latitude,
            "Restaurant Name": restaurant_name,
            "Cuisines": cuisines,
            "City": city,
            "Country Code": country_code,
            "Currency": currency,
            "Has Table booking": table_booking,
            "Has Online delivery": online_delivery,
        }
    ])

    prediction = model.predict(input_data)[0]

    prediction = max(1.0, min(5.0, prediction))

    return round(float(prediction), 2)


if __name__ == "__main__":

    rating = predict_rating(
        restaurant_name="Le Petit Souffle",
        cuisines="French, Japanese, Desserts",
        votes=314,
        average_cost=1100,
        price_range=3,
        city="Makati City",
        country_code=162,
        currency="Botswana Pula(P)",
        table_booking="Yes",
        online_delivery="No",
        longitude=121.027535,
        latitude=14.565443,
    )

    print(f"Predicted Restaurant Rating: {rating}/5.0")
