from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load trained model
MODEL_PATH = os.path.join("models", "crop_yield_model.joblib")
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


@app.route("/", methods=["GET"])
def home():
    return render_template("new.html", prediction_text=None, form_data=None)


@app.route("/predict", methods=["POST"])   # ✅ allow POST here
def predict():
    if model is None:
        return render_template(
            "new.html",
            prediction_text="Model not found. Please run train.py first.",
            form_data=request.form
        )

    # Get form data
    crop = request.form.get("crop")
    soil_type = request.form.get("soil_type")
    seed_variety = request.form.get("seed_variety")
    fertilizer = float(request.form.get("fertilizer", 0))
    area = float(request.form.get("area_hectares", 0))

    # Build dataframe with the same feature names used during training
    X = pd.DataFrame([{
        "crop": crop,
        "soil_type": soil_type,
        "seed_variety": seed_variety,
        "fertilizer": fertilizer,
        "area_hectares": area
    }])

    # Predict
    pred = model.predict(X)[0]
    prediction_text = f"🌾 Predicted yield for {crop}: {pred:.2f} (units same as training data)"

    return render_template("new.html", prediction_text=prediction_text, form_data=request.form)


if __name__ == "__main__":
    app.run(debug=True)
