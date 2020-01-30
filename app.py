import json
import sys
from flask import Flask, jsonify, request, render_template, url_for
from sklearn.externals import joblib
import numpy as np
import pickle

loaded_model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def main():
    """ Main page of the API """
    return render_template('index.html')
    
@app.route('/predict', methods=['GET'])
def predict():
    args = request.args
    required_args = ["produk","merek","tipe","ukuran"]
    # Simple error handling for the arguments
    diff = set(required_args).difference(set(args.keys()))
    person_features = np.array([produk_mapping[args["produk"]],
                                merek_mapping[args["merek"]],
                                tipe_mapping[args["tipe"]],
								ukuran_mapping[args["ukuran"]]
                               ])                                
    probability = model.predict(person_features)
    return render_template('index.html',output=probability[0])
    
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
