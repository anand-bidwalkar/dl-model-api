from flask import Flask, request
import tensorflow as tf
import os
import joblib

app = Flask(__name__)

real_estate_modelfile = 'model/real-estate'
real_estate_model=tf.keras.models.load_model(real_estate_modelfile)

aqi_modelfile = 'model/aqi/SVM_Final.pkl'
aqi_model = joblib.load(aqi_modelfile)

@app.route('/', methods=['POST'])
def post():    
    return "Welcome to D2I Geelong City Web API"


@app.route('/api/', methods=['POST'])
def makecalc():
    data = request.get_json()
    prediction = str(real_estate_model.predict(data)[0][0])
    return {'Result': prediction}

@app.route('/aqi_prediction/', methods=['POST'])
def make_prediction():
    data = request.get_json()
    prediction = str(aqi_model.predict(data)[0])
    return {'Result': prediction}

if __name__ == '__main__':
    app.run(debug=True)