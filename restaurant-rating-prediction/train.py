
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "Dataset.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. LOAD AND CLEAN DATASET
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print(f"Original records: {len(df)}")

target = "Aggregate rating"

# Remove restaurants without valid ratings
df[target] = pd.to_numeric(df[target], errors="coerce")

df = df.dropna(subset=[target])
df = df[df[target].between(1, 5)]

print(f"Usable records: {len(df)}")


# --------------------------------------------------
# 3. SELECT FEATURES
# --------------------------------------------------

numeric_features = [
    "Votes",
    "Average Cost for two",
    "Price range",
    "Longitude",
    "Latitude",
]

categorical_features = [
    "Restaurant Name",
    "Cuisines",
    "City",
    "Country Code",
    "Currency",
    "Has Table booking",
    "Has Online delivery",
]

features = numeric_features + categorical_features

X = df[features].copy()
y = df[target].copy()

# Convert numeric columns safely
for column in numeric_features:
    X[column] = pd.to_numeric(
        X[column], errors="coerce"
    )


# --------------------------------------------------
# 4. SPLIT DATA INTO TRAINING, VALIDATION AND TESTING
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

X_train, X_validation, y_train, y_validation = (
    train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        random_state=42,
    )
)

print("\nData split completed.")
print(f"Training records: {len(X_train)}")
print(f"Validation records: {len(X_validation)}")
print(f"Testing records: {len(X_test)}")


# --------------------------------------------------
# 5. BUILD DATA PREPROCESSING PIPELINE
# --------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=5,
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        (
            "categorical",
            categorical_transformer,
            categorical_features,
        ),
    ]
)


# --------------------------------------------------
# 6. DEFINE MACHINE LEARNING MODELS
# --------------------------------------------------

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    ),
}


# --------------------------------------------------
# 7. TRAIN AND COMPARE MODELS
# --------------------------------------------------

results = {}
best_model_name = None
best_pipeline = None
best_validation_mae = float("inf")

print("\n========== MODEL COMPARISON ==========")

for name, algorithm in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", algorithm),
        ]
    )

    print(f"\nTraining {name}...")

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_validation)

    mae = mean_absolute_error(
        y_validation, predictions
    )

    rmse = np.sqrt(
        mean_squared_error(y_validation, predictions)
    )

    r2 = r2_score(y_validation, predictions)

    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    if mae < best_validation_mae:
        best_validation_mae = mae
        best_model_name = name
        best_pipeline = pipeline


# --------------------------------------------------
# 8. REFIT THE BEST MODEL
# --------------------------------------------------

print("\n======================================")
print(f"Best model: {best_model_name}")
print("Retraining using training + validation data...")

X_combined = pd.concat([X_train, X_validation])
y_combined = pd.concat([y_train, y_validation])

best_pipeline.fit(X_combined, y_combined)


# --------------------------------------------------
# 9. FINAL MODEL EVALUATION
# --------------------------------------------------

test_predictions = best_pipeline.predict(X_test)

mae = mean_absolute_error(y_test, test_predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, test_predictions)
)

r2 = r2_score(y_test, test_predictions)

within_half = np.mean(
    np.abs(y_test.to_numpy() - test_predictions) <= 0.5
) * 100

print("\n========== FINAL TEST RESULTS ==========")
print(f"Selected model: {best_model_name}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")
print(f"Within 0.5 rating points: {within_half:.2f}%")
print("========================================")


# --------------------------------------------------
# 10. SAVE BEST MODEL
# --------------------------------------------------

model_path = MODEL_DIR / "rating_predictor.joblib"

joblib.dump(best_pipeline, model_path)

print(f"\nBest model saved to: {model_path}")
print("\nModel training completed successfully!")
