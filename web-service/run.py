# https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4

import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)
model = pickle.load(open('LogReg.pkl', 'rb'))


@app.route('/start', methods=['GET'])
def start():
    return 'Application is running!'


@app.route('/results', methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(str(output))


if __name__ == "__main__":
    app.run(debug=True)

