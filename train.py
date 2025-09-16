# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib
from pathlib import Path

# ✅ Load dataset
DATA_PATH = Path("data/crop_data.csv")
df = pd.read_csv(DATA_PATH)

# ✅ Normalize column names (strip spaces, lowercase)
df.columns = df.columns.str.strip().str.lower()

# ✅ Ensure 'yield' column exists
if "yield" not in df.columns:
    raise ValueError("❌ Dataset must have a 'Yield' column.")

# ✅ Features & target
X = df.drop("yield", axis=1)
y = df["yield"]

# ✅ Define categorical and numeric columns
categorical_cols = [col for col in X.columns if X[col].dtype == "object"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# ✅ Preprocessor
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols)
])

# ✅ Pipeline with RandomForest
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=200, random_state=42))
])

# ✅ Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Fit model
pipeline.fit(X_train, y_train)

# ✅ Save trained model
joblib.dump(pipeline, "crop_model.pkl")

print("✅ Model trained and saved as crop_model.pkl")
                