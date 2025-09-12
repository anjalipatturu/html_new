from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Path to saved model
MODEL_PATH = os.path.join("models", "crop_yield_model.joblib")

# Load the trained model if it exists
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


@app.route("/", methods=["GET"])
def home():
    # Always render the new.html template
    return render_template("new.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "⚠️ Model not found. Run train.py first."}), 500

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid request. Expecting JSON."}), 400

    try:
        crop = data.get("crop")
        soil_type = data.get("soil_type")
        seed_variety = data.get("seed_variety")
        fertilizer = float(data.get("fertilizer", 0))
        area_hectares = float(data.get("area_hectares", 0))

        # Create input dataframe
        X = pd.DataFrame([{
            "crop": crop,
            "soil_type": soil_type,
            "seed_variety": seed_variety,
            "fertilizer": fertilizer,
            "area_hectares": area_hectares
        }])

        pred = model.predict(X)[0]
        return jsonify({
            "prediction": float(pred),
            "message": f"🌾 Predicted yield for {crop}: {pred:.2f}"
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
