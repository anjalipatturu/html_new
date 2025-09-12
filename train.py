import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

def print_metrics(y_true, y_pred, prefix=""):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{prefix} RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.3f}")

def main():
    # Load dataset
    df = pd.read_csv("data/crop_data.csv")

    # Clean column names
    df.columns = [c.strip().lower() for c in df.columns]

    numeric_features = ["fertilizer", "area_hectares"]
    categorical_features = ["soil_type", "seed_variety"]
    target = "yield"

    expected = numeric_features + categorical_features + [target]
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing columns: {missing}. Found columns: {list(df.columns)}")

    X = df[numeric_features + categorical_features]
    y = df[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing with polynomial features for numeric
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False))
        ]), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])

    # Model pipeline with RandomForest
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=42))
    ])

    # Baseline training
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print_metrics(y_test, y_pred, prefix="Baseline")

    # Grid search tuning
    param_grid = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
    grid.fit(X_train, y_train)

    print("✅ Best params:", grid.best_params_)
    best_model = grid.best_estimator_

    # Evaluate tuned model
    y_pred_best = best_model.predict(X_test)
    print_metrics(y_test, y_pred_best, prefix="Tuned")

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/crop_yield_model.joblib")
    print("✅ Model saved to models/crop_yield_model.joblib")

if __name__ == "__main__":
    main()
