import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset
df = pd.read_csv("data/crop_data.csv")

# Standardize column names
df.columns = [c.lower().strip() for c in df.columns]
df = df.rename(columns={"crop":"crop_name","soil":"soil_type","seed":"seed_variety"})

# Fill missing numeric values
num_cols = ["fertilizer_kg", "no_of_acres", "yield"]
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

# Fill missing categorical values
cat_cols = ["crop_name","soil_type","seed_variety"]
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

# Derived features
df["fertilizer_per_acre"] = df["fertilizer_kg"] / df["no_of_acres"]

# Features and target
X = df[["crop_name","soil_type","seed_variety","fertilizer_kg","no_of_acres","fertilizer_per_acre"]]
y = df["yield"]

# Preprocessing: categorical + numeric scaling
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["crop_name","soil_type","seed_variety"]),
        ("num", StandardScaler(), ["fertilizer_kg","no_of_acres","fertilizer_per_acre"])
    ]
)

# Pipeline with HistGradientBoostingRegressor
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", HistGradientBoostingRegressor(
        max_iter=3000,
        max_depth=25,
        learning_rate=0.05,
        min_samples_leaf=3,
        random_state=42
    ))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ RMSE: {rmse:.2f} kg")
print(f"✅ R2 Score: {r2:.2f}")

# Save model
with open("crop_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model trained and saved as crop_model.pkl")
