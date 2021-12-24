import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import trs

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [str(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    prediction = trs.predict(int_features)
    return render_template('index.html', prediction_text=prediction)

if __name__ == "__main__":
    app.run(debug=True)

    