from fastapi import FastAPI
import pickle
from pandas import DataFrame
import json
import numpy as np
import lightgbm as lgb

app = FastAPI()

@app.get("/predict")
def predict(data):
    model = pickle.load(open('pickles/pp_pred_pipe_gbc_new1.pkl', 'rb'))
    param_data = json.loads(data)
    X_pred = DataFrame(param_data,index=[0]).replace('',np.nan)

    def alt_nan_imp(X):

        cat_vars = ['B_30',
            'B_38',
            'D_114',
            'D_116',
            'D_117',
            'D_120',
            'D_126',
            'D_63',
            'D_64',
            'D_66',
            'D_68']

        alt_nan_list = [-1,-1.0, "-1.0", "-1"]

        cat_columns = [column for column in X.columns if column in cat_vars]

        X[cat_columns] = X[cat_columns].applymap(lambda x: np.nan if x in alt_nan_list else x)

    alt_nan_imp(X_pred)

    prediction = model.predict(X_pred)[0]
    pred_probability = model.predict_proba(X_pred)

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
