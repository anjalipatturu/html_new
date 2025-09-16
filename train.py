# train.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

# ------------------------------
# Load dataset
# ------------------------------
DATA_PATH = Path("data/crop_data.csv")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# Standardize column names
df.columns = [c.strip().lower() for c in df.columns]

# Expected base columns
rename_map = {
    "crop": "crop_name",
    "soil": "soil_type",
    "seed": "seed_variety",
    "fertilizer": "fertilizer_kg",
    "acres": "no_of_acres"
}
df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

# ------------------------------
# Detect target (production / yield)
# ------------------------------
possible_targets = ["production", "yield", "yield_kg", "yield (quintals)", "output"]
target_col = None
for col in df.columns:
    if col.lower() in possible_targets:
        target_col = col
        break

if not target_col:
    raise KeyError(
        f"No production/yield column found in dataset. Available columns: {df.columns}"
    )

df = df.rename(columns={target_col: "production"})

# ------------------------------
# Convert production to kilograms
# ------------------------------
if df["production"].max() < 50:
    # assume tons → convert to kg
    df["production_kg"] = df["production"] * 1000
elif df["production"].max() < 500:
    # assume quintals → convert to kg
    df["production_kg"] = df["production"] * 100
else:
    # already in kg
    df["production_kg"] = df["production"]

# ------------------------------
# Feature engineering
# ------------------------------
df["fertilizer_per_acre"] = df["fertilizer_kg"] / df["no_of_acres"].replace(0, 1)

# Features & Target
X = df[["crop_name", "soil_type", "seed_variety",
        "fertilizer_kg", "no_of_acres", "fertilizer_per_acre"]]
y = df["production_kg"]

# ------------------------------
# Preprocessing
# ------------------------------
categorical_cols = ["crop_name", "soil_type", "seed_variety"]
numeric_cols = ["fertilizer_kg", "no_of_acres", "fertilizer_per_acre"]

categorical_transformer = OneHotEncoder(handle_unknown="ignore")
numeric_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", categorical_transformer, categorical_cols),
        ("numeric", numeric_transformer, numeric_cols),
    ]
)

# ------------------------------
# Model pipeline
# ------------------------------
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=300, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
pipeline.fit(X_train, y_train)

# ------------------------------
# Save model
# ------------------------------
MODEL_PATH = Path("crop_model.pkl")
joblib.dump(pipeline, MODEL_PATH)

print(f"✅ Model trained and saved to {MODEL_PATH}")
print("🔍 Sample predictions:")
print(pipeline.predict(X_test[:5]))
