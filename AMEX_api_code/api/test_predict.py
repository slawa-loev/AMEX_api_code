import pandas as pd
import pickle
import json
import numpy as np

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

# def predict(data):
#     model = pickle.load(open('pickles/pp_pred_pipe_gbc_ws.pkl', 'rb'))
#     param_data = json.loads(data)
#     X_pred = DataFrame(param_data,index=[0]).replace('',np.nan)

    # prediction = model.predict(transform_X)[0]
    # pred_probability = model.predict_proba(transform_X)

    # if prediction == 1:
    #     defaulter = 'defaulter'
    # else:
    #     defaulter = 'payer'

    # return {'customer_ID':param_data['customer_ID'],
    #         'output':defaulter,
    #         'probability':round(pred_probability[0][1],3)}


# data = pd.read_csv('AMEX_api_code/api/test_row.csv',index_col=0).to_dict(orient='records')[0]
# # print(data)
# # param_dict = {"data": json.dumps(params)}
# print(predict(data))


data = pd.read_csv('AMEX_api_code/api/test_row.csv',index_col=0).fillna('').to_dict(orient='records')[0]
# data = {'Unnamed: 0': 0, 'customer_ID': '6ba461c93869797c49b0f34c29274e50915466eda02a82fde4aa2c73c3924339', 'S_2': '2017-04-29', 'P_2': 0.5243904931830153, 'D_39': 0.3313523183745179, 'B_1': 0.0226474075652035, 'B_2': 1.0098138176705274, 'R_1': 0.0094418612623092, 'S_3': 0.3013753140016364, 'D_41': 0.0078785629732383, 'B_3': 0.0258205059627975, 'D_42': 0.1442666865165377, 'D_43': 0.2719021428514478, 'D_44': nan, 'B_4': 0.0283137155569896, 'D_45': 0.0181600179345101, 'B_5': 0.042604957206283, 'R_2': 0.0076086225667935, 'D_46': 0.4423322496494613, 'D_47': 0.5640947570546344, 'D_48': nan, 'D_49': nan, 'B_6': 0.1141169198300304, 'B_7': 0.0364595248656915, 'B_8': 1.001206164416357, 'D_50': nan, 'D_51': 0.0055727585123258, 'B_9': 0.2995098902026097, 'R_3': 0.5078069824461213, 'D_52': 0.0065321521685339, 'P_3': 0.9704147394434446, 'B_10': 0.1931363807984167, 'D_53': nan, 'S_5': 0.0497382032113571, 'B_11': 0.0170887078387576, 'S_6': 6.264821241336181e-06, 'D_54': 1.003239737456082, 'R_4': 0.0097192267864564, 'S_7': 0.5836650003043637, 'B_12': 0.0279422840289572, 'S_8': 0.4724606934230521, 'D_55': 0.3718658693500563, 'D_56': nan, 'B_13': 0.0229769009343403, 'R_5': 0.0024600658719739, 'D_58': 0.092685414392393, 'S_9': nan, 'B_14': 0.0187415315595689, 'D_59': 0.2725730447049424, 'D_60': 0.5128522688445263, 'D_61': 0.996966160590678, 'B_15': 0.008916801024969, 'S_11': 0.2839718258314885, 'D_62': 0.0063765535043084, 'D_63': 'CO', 'D_64': 'U', 'D_65': 0.0082769054925904, 'B_16': 0.3406488182369175, 'B_17': 0.009998703087497, 'B_18': 0.6902283738451007, 'B_19': 0.0081695078433936, 'D_66': nan, 'B_20': 0.0006330919917411, 'D_68': 3.0, 'S_12': 0.1916605915252538, 'R_6': 0.0080456630109759, 'S_13': 0.0018278306292617, 'B_21': 0.0013618570248982, 'D_69': 0.0985397767929114, 'B_22': 0.0072234345795886, 'D_70': 0.0090138733828583, 'D_71': 0.0069688305279365, 'D_72': 0.341570684555522, 'S_15': 0.2066711654452798, 'B_23': 0.0211564693339486, 'D_73': nan, 'P_4': 0.9268010960381086, 'D_74': 0.0806280135617195, 'D_75': 0.0729107856856167, 'D_76': nan, 'B_24': 0.0007743527406967, 'R_7': 0.0036305356399982, 'D_77': nan, 'B_25': 0.0269237456259648, 'B_26': 0.0020041265394678, 'D_78': nan, 'D_79': 0.0019684030320729, 'R_8': 0.0095986871190017, 'R_9': nan, 'S_16': 0.0073472057447631, 'D_80': 0.0046132681510072, 'R_10': 0.0002366237215068, 'R_11': 0.5030766236950268, 'B_27': 0.0080865243484111, 'D_81': 0.0016620748567789, 'D_82': nan, 'S_17': 0.0080248355218849, 'R_12': 1.0088355475409474, 'B_28': 0.0174737864388395, 'R_13': 0.005137948786245, 'D_83': 1.0010212222511732, 'R_14': 0.0097658691950037, 'R_15': 0.007744656789712, 'D_84': 0.0053338538809249, 'R_16': 0.5068937771309112, 'B_29': nan, 'B_30': 0.0, 'S_18': 0.0025881480646735, 'D_86': 0.0084592043500856, 'D_87': nan, 'R_17': 0.0064107803599046, 'R_18': 0.0021582068627253, 'D_88': nan, 'B_31': 1, 'S_19': 0.0036599241820819, 'R_19': 0.0023951620876795, 'B_32': 0.0058979996465047, 'S_20': 0.0084387973217704, 'R_20': 0.0072765937081887, 'R_21': 0.0026307474714683, 'B_33': 1.0042078645998118, 'D_89': 0.0093799559198365, 'R_22': 0.0032245356122126, 'R_23': 0.0084210175139439, 'D_91': 0.0021117280992512, 'D_92': 0.0002813886044537, 'D_93': 0.0064740428785274, 'D_94': 0.0034891282055373, 'R_24': 0.0074637673580497, 'R_25': 8.514154457408552e-05, 'D_96': 0.0022346601508937, 'S_22': 0.9896390752145902, 'S_23': 0.1321254122831956, 'S_24': 0.9834439806950838, 'S_25': 0.9738907537358952, 'S_26': 0.0038862860431581, 'D_102': 0.289217832863587, 'D_103': 1.0056231504565114, 'D_104': 0.9613147105437548, 'D_105': 0.1697762859187989, 'D_106': nan, 'D_107': 0.3415914097814573, 'B_36': 0.0014437035216303, 'B_37': 0.0269399028855463, 'R_26': 0.1079921756486303, 'R_27': 0.0102809017375042, 'B_38': 2.0, 'D_108': nan, 'D_109': 0.0045981590747289, 'D_110': nan, 'D_111': nan, 'B_39': nan, 'D_112': 0.0126469786018308, 'B_40': 0.111658105064575, 'S_27': 0.0085330483574145, 'D_113': 0.6060588458397328, 'D_114': 0.0, 'D_115': 0.0543636921559064, 'D_116': 0.0, 'D_117': 3.0, 'D_118': 0.1168077296764655, 'D_119': 0.1096212138082288, 'D_120': 0.0, 'D_121': 0.650981600072707, 'D_122': 0.146803227388556, 'D_123': 0.0011049696432874, 'D_124': 0.8183435054999706, 'D_125': 0.000647677346751, 'D_126': 1.0, 'D_127': 5.266267486242482e-06, 'D_128': 0.0029830598258818, 'D_129': 0.007395236340952, 'B_41': 0.0031757932139252, 'B_42': nan, 'D_130': 1.0020848849400965, 'D_131': 0.0061371572650052, 'D_132': nan, 'D_133': 0.0018016269009619, 'R_28': 0.0021418392238124, 'D_134': nan, 'D_135': nan, 'D_136': nan, 'D_137': nan, 'D_138': nan, 'D_139': 1.0080621413676547, 'D_140': 0.0035948604672395, 'D_141': 0.8744796412184572, 'D_142': 0.0901836791309312, 'D_143': 1.0018106904900153, 'D_144': 0.0044457500675465, 'D_145': 0.3653796067325561}
params = {"data": json.dumps(data)}
#response = requests.get(url, params=params)

def predict(data):
    model = pickle.load(open('pickles/pp_pred_pipe_gbc_new.pkl', 'rb'))
    param_data = json.loads(data['data'])
    X_pred = pd.DataFrame(param_data,index=[0]).replace('',np.nan)

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
    else:
        defaulter = 'payer'

    return {'customer_ID':param_data['customer_ID'],
            'output':defaulter,
            'probability':round(pred_probability[0][1],3)}

print(predict(params))
