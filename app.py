from flask import Flask, request
import tensorflow as tf
import os

app = Flask(__name__)


@app.route('/api/', methods=['POST'])
def makecalc():
    data = request.get_json()
    prediction = str(model.predict(data)[0][0])
    return {'Result': prediction}

if __name__ == '__main__':
    modelfile = 'model/'
    model=tf.keras.models.load_model(modelfile)
    app.run(debug=True)