from flask import Flask, render_template, request
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)

MODEL_PATH = Path("crop_model.pkl")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Trained model not found. Run train.py to create {MODEL_PATH}")

pipeline = joblib.load(MODEL_PATH)

# Helper to build input dataframe
def build_input_df(crop_name, soil_type, seed_variety, fertilizer_kg, no_of_acres):
    crop_name = str(crop_name).strip().title()
    soil_type = str(soil_type).strip().title()
    seed_variety = str(seed_variety).strip().title()
    fertilizer_kg = float(fertilizer_kg)
    no_of_acres = float(no_of_acres)
    fertilizer_per_acre = fertilizer_kg / no_of_acres if no_of_acres > 0 else 0
    return pd.DataFrame({
        "crop_name": [crop_name],
        "soil_type": [soil_type],
        "seed_variety": [seed_variety],
        "fertilizer_kg": [fertilizer_kg],
        "no_of_acres": [no_of_acres],
        "fertilizer_per_acre": [fertilizer_per_acre]
    })

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    suggestions = None
    form = request.form
    if request.method == "POST":
        if form.get("action") == "clear":
            form = {}
        else:
            crop_name = form.get("crop_name", "Wheat")
            soil_type = form.get("soil_type", "Loamy")
            fertilizer_kg = float(form.get("fertilizer_kg", 50) or 50)
            no_of_acres = float(form.get("no_of_acres", 1) or 1)
            seed_variety = form.get("seed_variety", "Hybrid")

            X = build_input_df(crop_name, soil_type, seed_variety, fertilizer_kg, no_of_acres)

            if form.get("action") == "predict":
                y_pred = pipeline.predict(X)[0]
                result = f"🌱 Predicted Yield: {int(round(y_pred)):,} kg"

            elif form.get("action") == "optimize":
                fert_min = max(0, int(fertilizer_kg - 50))
                fert_max = min(1000, int(fertilizer_kg + 50))
                fert_candidates = list(range(fert_min, fert_max + 1, 10))
                seed_options = ["Hybrid", "Improved", "Local"]

                best = {"pred": -1e9, "fert": None, "seed": None}
                for s in seed_options:
                    for f in fert_candidates:
                        X_try = build_input_df(crop_name, soil_type, s, f, no_of_acres)
                        pred = pipeline.predict(X_try)[0]
                        if pred > best["pred"]:
                            best = {"pred": pred, "fert": f, "seed": s}

                suggestions = []
                current_pred = pipeline.predict(X)[0]
                if best["pred"] > current_pred + 1e-6:
                    gain = best["pred"] - current_pred
                    suggestions.append(
                        f"✅ Best found: Use {best['seed']} seeds with {best['fert']} kg fertilizer → predicted yield {int(round(best['pred'])):,} kg (gain {int(round(gain)):,} kg)."
                    )
                    suggestions.append("Note: Verify with local agronomic guidance before large changes.")
                else:
                    suggestions.append("✅ Current inputs are near-optimal in this search range.")

                if fertilizer_kg < 50:
                    suggestions.append("💡 Soil tests help tune fertilizer — low fertilizer may limit yield.")
                if seed_variety.lower() != "hybrid":
                    suggestions.append("💡 Hybrid seeds often increase yield (but check cost/availability).")
                suggestions.append("ℹ️ These are model-based suggestions. Consult local experts for field-level advice.")

                result = f"🌱 Current predicted yield: {int(round(current_pred)):,} kg | Best candidate predicted: {int(round(best['pred'])):,} kg"

    return render_template("new.html", result=result, form=form, suggestions=suggestions)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
