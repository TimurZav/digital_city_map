import os
import pickle
from flask_cors import CORS
from flask import Flask, request
from pandas import json_normalize

app: Flask = Flask(__name__)
CORS(app)

@app.post("/")
def get_house_prediction_price():
    response = request.json
    dataset = json_normalize(response)
    model_file = "model_house_prediction_price.sav"
    if os.path.isfile(model_file):
        loaded_model = pickle.load(open(model_file, 'rb'))
        return str(round(loaded_model.predict(dataset)[0], 2))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)