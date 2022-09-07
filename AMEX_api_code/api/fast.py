from fastapi import FastAPI
import pickle
from pandas import DataFrame
import json
import numpy as np
import catboost as ctb
from AMEX_api_code.data.data import data_agg, get_difference, feat_eng

app = FastAPI()

@app.get("/predict")
def predict(data):
    model = pickle.load(open('pickles/cat_boost_reloaded_079', 'rb'))
    param_data = json.loads(data)
    X_pred = DataFrame(param_data,index=[0]).replace('',np.nan)

    X_pred_agg = data_agg(X_pred).drop(columns=['customer_ID'])

    prediction = model.predict(X_pred_agg)[0]
    pred_probability = model.predict_proba(X_pred_agg)

    if prediction == 1:
        defaulter = 'defaulter'
        probability = round(pred_probability[0][1],3)
    else:
        defaulter = 'payer'
        probability = round(pred_probability[0][0],3)

    return {'customer_ID':param_data['customer_ID'],
            'output':defaulter,
            'probability':probability}


@app.get("/")
def root():
    return {'customer_ID':'abc123customer',
            'output': 'defaulter',
            'probability': '0.999'}
