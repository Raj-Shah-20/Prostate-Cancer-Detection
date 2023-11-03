import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
loaded_model = pickle.load(open('finalized_model.sav','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    s = np.load('std.npy')
    m = np.load('mean.npy')
    int_features = [float(x) for x in request.form.values()]
    final_features = (np.array([int_features]-m))/s
    predicted = loaded_model.predict(final_features)
    return render_template('output.html', data=predicted)

if __name__ == "__main__":
    app.run(debug=True)