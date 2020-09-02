# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 13:55:57 2020

@author: Admin
"""
import pandas as pd
import numpy as np
from flask import Flask, request,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    feature_name =['Term','NoEmp','CreateJob','RetainedJob','UrbanRural','FranchiseStatus',
                   'NewExist','RevLineCr','LowDoc','DisbursementGross']
    
    df = pd.DataFrame(final_features,columns=feature_name)
    output= model.predict(df)
    
    if output ==1:
        res_val = "*Not_Default*"
    else:
        res_val = "*Default*"

    return render_template('index.html', prediction_text='Loan Status : {}'.format(res_val))


if __name__ == "__main__":
    app.run(debug=True)
