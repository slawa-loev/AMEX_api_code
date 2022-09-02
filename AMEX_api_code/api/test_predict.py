import pandas as pd
import pickle
import json

# X_pred = pd.read_csv('AMEX_api_code/api/test_row.csv')
# def load_model():
#     model = pickle.load(open('pickles/pp_pred_pipe', 'rb'))
#     return model

# print(load_model().predict(X_pred))

# def predict(params):
#     model = pickle.load(open('pickles/pp_pred_pipe', 'rb'))
#     X_pred = pd.DataFrame.from_dict(params,orient='index').transpose()
#     prediction = model.predict(X_pred)[0]
#     pred_probability = model.predict_proba(X_pred)

#     if prediction == 1:
#         defaulter = 'defaulter'
#     else:
#         defaulter = 'payer'

#     return {'customer_ID':params['customer_ID'],
#             'output':defaulter,
#             'probability':round(pred_probability[0][1],3)}


# def predict(data):
#     model = pickle.load(open('pickles/pp_pred_pipe_gbc_ws', 'rb'))

#     X_pred = pd.DataFrame.from_dict(data,orient='index').transpose()
#     prediction = model.predict(X_pred)[0]
#     pred_probability = model.predict_proba(X_pred)

#     if prediction == 1:
#         defaulter = 'defaulter'
#     else:
#         defaulter = 'payer'

#     return {'customer_ID':param_data['customer_ID'],
#             'output':defaulter,
#             'probability':round(pred_probability[0][1],3)}

def predict(data):
    model = pickle.load(open('pickles/pp_pred_pipe_gbc_ws.pkl', 'rb'))
    param_data = json.loads(data)
    X_pred = DataFrame(param_data,index=[0]).replace('',np.nan)

    # def nan_imp(X):
    #     nan_list = [-1,-1.0, "-1.0", "-1"]
    #     return X.applymap(lambda x: np.nan if x in nan_list else x)

    # nan_imp(X_pred)
    transform_X = model.transform(X_pred)
    # prediction = model.predict(transform_X)[0]
    # pred_probability = model.predict_proba(transform_X)

    # if prediction == 1:
    #     defaulter = 'defaulter'
    # else:
    #     defaulter = 'payer'

    # return {'customer_ID':param_data['customer_ID'],
    #         'output':defaulter,
    #         'probability':round(pred_probability[0][1],3)}
    return transform_X

data = pd.read_csv('AMEX_api_code/api/test_row.csv',index_col=0).to_dict(orient='records')[0]
# print(data)
# param_dict = {"data": json.dumps(params)}
print(predict(data))
