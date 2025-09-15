from flask import Flask, render_template_string, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
with open("crop_model.pkl", "rb") as f:
    model = pickle.load(f)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>🌾 Crop Yield Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
<style>
    body {
        font-family: 'Roboto', sans-serif;
        background: #ffffff; /* Changed to pure white */
        margin: 0;
        padding: 0;
    }
    h2 {
        text-align: center;
        color: #034f84;
        margin-top: 40px;
        font-weight: 700;
        font-size: 2.2em;
    }
    form {
        width: 450px;
        margin: 40px auto;
        padding: 30px;
        background: #ffffffcc;
        border-radius: 20px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.2);
        transition: 0.3s;
    }
    form:hover { box-shadow: 0 18px 40px rgba(0,0,0,0.3); }
    label {
        font-weight: 500;
        margin-top: 10px;
        display: block;
        color: #034f84;
    }
    input, select {
        width: 100%;
        padding: 12px;
        margin: 8px 0 20px 0;
        border-radius: 12px;
        border: 1px solid #ccc;
        box-sizing: border-box;
        font-size: 15px;
        transition: 0.3s;
    }
    input:focus, select:focus {
        border-color: #0572b5;
        box-shadow: 0 0 5px rgba(5,114,181,0.5);
        outline: none;
    }
    .button-container {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
    }
    button {
        width: 30%;
        padding: 12px;
        margin: 5px 0;
        background: linear-gradient(45deg, #034f84, #0572b5);
        color: #fff;
        border: none;
        border-radius: 12px;
        cursor: pointer;
        font-weight: 600;
        transition: 0.3s;
    }
    button:hover {
        background: linear-gradient(45deg, #0572b5, #034f84);
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
    }
    .result-card {
        width: 450px;
        margin: 20px auto;
        padding: 20px;
        background: #f8f8f8;
        border-radius: 15px;
        text-align: center;
        font-weight: 600;
        color: #034f84;
        font-size: 1.3em;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        transition: 0.3s;
    }
    .result-card:hover {
        box-shadow: 0 12px 30px rgba(0,0,0,0.2);
    }
    @media(max-width:500px){
        form, .result-card { width: 90%; padding: 20px; }
        button { width: 45%; margin: 5px 2%; }
    }
</style>
</head>
<body>
<h2>🌾 Crop Yield Prediction & Optimization</h2>
<form method="POST">
    <label>Crop Name:</label>
    <select name="crop_name">
        <option {{'selected' if form.get('crop_name')=='Wheat' else ''}}>Wheat</option>
        <option {{'selected' if form.get('crop_name')=='Rice' else ''}}>Rice</option>
        <option {{'selected' if form.get('crop_name')=='Maize' else ''}}>Maize</option>
        <option {{'selected' if form.get('crop_name')=='Soybean' else ''}}>Soybean</option>
    </select>

    <label>Soil Type:</label>
    <select name="soil_type">
        <option {{'selected' if form.get('soil_type')=='Clay' else ''}}>Clay</option>
        <option {{'selected' if form.get('soil_type')=='Sandy' else ''}}>Sandy</option>
        <option {{'selected' if form.get('soil_type')=='Loamy' else ''}}>Loamy</option>
        <option {{'selected' if form.get('soil_type')=='Silty' else ''}}>Silty</option>
    </select>

    <label>Fertilizer (kg):</label>
    <input type="number" name="fertilizer_kg" value="{{form.get('fertilizer_kg','50')}}" step="1"/>

    <label>No. of Acres:</label>
    <input type="number" name="no_of_acres" value="{{form.get('no_of_acres','1')}}" step="0.1"/>

    <label>Seed Variety:</label>
    <select name="seed_variety">
        <option {{'selected' if form.get('seed_variety')=='Hybrid' else ''}}>Hybrid</option>
        <option {{'selected' if form.get('seed_variety')=='Local' else ''}}>Local</option>
        <option {{'selected' if form.get('seed_variety')=='Improved' else ''}}>Improved</option>
    </select>

    <div class="button-container">
        <button type="submit" name="action" value="predict">Predict</button>
        <button type="submit" name="action" value="optimize">Optimize</button>
        <button type="submit" name="action" value="clear">Clear</button>
    </div>
</form>

{% if result %}
<div class="result-card">{{result}}</div>
{% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET","POST"])
def home():
    result = ""
    form = request.form
    if request.method == "POST":
        if form.get("action") == "clear":
            form = {}
        else:
            crop_name = form.get("crop_name", "Wheat")
            soil_type = form.get("soil_type", "Loamy")
            fertilizer_kg = float(form.get("fertilizer_kg", 50))
            no_of_acres = float(form.get("no_of_acres", 1))
            seed_variety = form.get("seed_variety", "Hybrid")

            fertilizer_per_acre = fertilizer_kg / max(no_of_acres, 0.01)

            # Input DataFrame
            X = pd.DataFrame([[crop_name, soil_type, seed_variety, fertilizer_kg, no_of_acres, fertilizer_per_acre]],
                             columns=["crop_name","soil_type","seed_variety","fertilizer_kg","no_of_acres","fertilizer_per_acre"])

            if form.get("action") == "predict":
                y_pred = model.predict(X)[0]
                y_pred = max(100, y_pred)  # Ensure three-digit kg
                result = f"🌱 Predicted Yield: {y_pred:.2f} kg"

            elif form.get("action") == "optimize":
                suggestions = []
                if fertilizer_kg < 100:
                    suggestions.append(f"💡 Increase fertilizer from {fertilizer_kg} kg")
                if no_of_acres < 5:
                    suggestions.append(f"💡 Consider increasing area from {no_of_acres} acres")
                if seed_variety.lower() != "hybrid":
                    suggestions.append("💡 Use Hybrid seeds for better yield")
                if not suggestions:
                    suggestions.append("✅ Inputs are optimal based on current model")
                result = " | ".join(suggestions)

    return render_template_string(HTML, result=result, form=form)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
