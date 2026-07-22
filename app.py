from flask import Flask, request, jsonify
from joblib import load
import pandas as pd

app = Flask(__name__)

# Load trained model
model = load("solar_model.pkl")

@app.route("/")
def home():
    return "Solar Power Prediction API is running."

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    input_df = pd.DataFrame([[
        data["IRRADIATION"],
        data["MODULE_TEMPERATURE"],
        data["AMBIENT_TEMPERATURE"],
        data["HOUR"],
        data["DAY"]
    ]], columns=[
        "IRRADIATION",
        "MODULE_TEMPERATURE",
        "AMBIENT_TEMPERATURE",
        "HOUR",
        "DAY"
    ])

    prediction = model.predict(input_df)[0]

    return jsonify({
        "predicted_dc_power": float(prediction)
    })

if __name__ == "__main__":
    app.run(debug=True)