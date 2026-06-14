from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

model = joblib.load("ml-model/traffic_model.pkl")

@app.route("/")
def home():
    return send_from_directory("../frontend", "index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    temp = float(data["temp"])
    rain = float(data["rain"])
    snow = float(data["snow"])
    clouds = float(data["clouds"])

    prediction = model.predict(
        np.array([[temp, rain, snow, clouds]])
    )

    return jsonify({
        "predicted_traffic": int(prediction[0])
    })

if __name__ == "__main__":
    app.run(debug=True)