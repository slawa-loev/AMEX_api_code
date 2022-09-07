import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

def get_difference(data, num_features):
    df1 = []
    customer_ids = []
    for customer_id, df in data.groupby(['customer_ID']):
        # Get the differences
        diff_df1 = df[num_features].diff(1).iloc[[-1]].values.astype(np.float32)
        # Append to lists
        df1.append(diff_df1)
        customer_ids.append(customer_id)
    # Concatenate
    df1 = np.concatenate(df1, axis = 0)
    # Transform to dataframe
    df1 = pd.DataFrame(df1, columns = [col + '_diff1' for col in df[num_features].columns])
    # Add customer id
    df1['customer_ID'] = customer_ids
    return df1

def feat_eng(df):

    ## adding some engineered features seen here https://www.kaggle.com/code/swimmy/tuffline-amex-anotherfeaturelgbm

    df["c_PD_239"]=df["D_39"]/(df["P_2"]*(-1)+0.0001)
    df["c_PB_29"]=df["P_2"]*(-1)/(df["B_9"]*(1)+0.0001)
    df["c_PR_21"]=df["P_2"]*(-1)/(df["R_1"]+0.0001)

    df["c_BBBB"]=(df["B_9"]+0.001)/(df["B_23"]+df["B_3"]+0.0001)
    df["c_BBBB1"]=(df["B_33"]*(-1))+(df["B_18"]*(-1)+df["S_25"]*(1)+0.0001)
    df["c_BBBB2"]=(df["B_19"]+df["B_20"]+df["B_4"]+0.0001)

    df["c_RRR0"]=(df["R_3"]+0.001)/(df["R_2"]+df["R_4"]+0.0001)
    df["c_RRR1"]=(df["D_62"]+0.001)/(df["D_112"]+df["R_27"]+0.0001)

    df["c_PD_348"]=df["D_48"]/(df["P_3"]+0.0001)
    df["c_PD_355"]=df["D_55"]/(df["P_3"]+0.0001)

    df["c_PD_439"]=df["D_39"]/(df["P_4"]+0.0001)
    df["c_PB_49"]=df["B_9"]/(df["P_4"]+0.0001)
    df["c_PR_41"]=df["R_1"]/(df["P_4"]+0.0001)

    return df

def data_agg(df, feat_eng=False): ## pass uploaded data

    ## add feature enginering if feat_eng True
    if feat_eng:
        df = feat_eng(df)

    ## get all feature names, except customer_ID and dates
    features = df.drop(['customer_ID', 'S_2'], axis = 1).columns.to_list()

    ## list of categorical features
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

    ## list of numerical features
    num_features = [feature for feature in features if feature not in cat_vars]

    train_num_agg = df.groupby("customer_ID")[num_features].agg(['mean', 'std', 'min', 'max', 'last']) # give summary statistics for each numerical feature
    train_num_agg.columns = ['_'.join(x) for x in train_num_agg.columns] # join the column name tuples to a single name
    train_num_agg.reset_index(inplace = True) # get the customer_ID in as a column again and reset index

    ## get lag difference data for numerical features
    train_diff = get_difference(df, num_features)

    ## categorical feature aggregation
    train_cat_agg = df.groupby("customer_ID")[cat_vars].agg(['count', 'last', 'nunique']) # give summary statistics for each categrocial feature
    train_cat_agg.columns = ['_'.join(x) for x in train_cat_agg.columns] # join the column name tuples to a single name
    train_cat_agg.reset_index(inplace = True) # get the customer_ID in as a column again and reset index

    ## merge dfs
    df_agg = train_num_agg.merge(train_cat_agg, how = 'inner', on = 'customer_ID').merge(train_diff, how = 'inner', on = 'customer_ID')

    ## ordinal encode cat_features
    cat_features = [f"{cf}_last" for cf in cat_vars]
    encoder = OrdinalEncoder()
    df_agg[cat_features] = encoder.fit_transform(df_agg[cat_features])

    ## add last - mean feature (only numerical features have means)
    num_cols_mean = [col for col in df_agg.columns if 'mean' in col]
    num_cols_last = [col for col in df_agg.columns if 'last' in col and col not in cat_features]

    for col in range(len(num_cols_last)):
        try:
            df_agg[f'{num_cols_last[col]}_mean_diff'] = df_agg[num_cols_last[col]] - df_agg[num_cols_mean[col]]
        except:
            pass

    return df_agg
