from flask import Flask, request
import tensorflow as tf
import os

app = Flask(__name__)
modelfile = 'model/'
model=tf.keras.models.load_model(modelfile)

@app.route('/', methods=['POST'])
def post():    
    return "Welcome to D2I Geelong City Web API"


@app.route('/api/', methods=['POST'])
def makecalc():
    data = request.get_json()
    prediction = str(model.predict(data)[0][0])
    return {'Result': prediction}

if __name__ == '__main__':
    app.run(debug=True)