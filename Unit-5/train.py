import pandas as pd
import numpy as np
# import wget
import pickle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Parameters of the model
C = 1.0
n_splits = 5
output_file = f'model_C=1.0.bin'

path = 'D:/Computacional/machine-learning-zoomcamp/Unit-5/'
# data = wget.download('https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df = pd.read_csv(path+'WA_Fn-UseC_-Telco-Customer-Churn.csv')

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0) # not the best solution

df.columns = df.columns.str.lower().str.replace(' ', '_')

string_columns = list(df.dtypes[df.dtypes == 'object'].index)

for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')

df.churn = (df.churn == 'yes').astype(int) # change yes & no for 1 & 0

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

# we need to split it one more time because it only separates in 2
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=11)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

del df_train['churn']
del df_val['churn']
del df_test['churn']

categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
               'phoneservice', 'multiplelines', 'internetservice',
               'onlinesecurity', 'onlinebackup', 'deviceprotection',
               'techsupport', 'streamingtv', 'streamingmovies',
               'contract', 'paperlessbilling', 'paymentmethod']
numerical = ['tenure', 'monthlycharges', 'totalcharges']

# Preprocessing + training of model
def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model

# Prediction of y values
def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

# Validation
print(f'doing validation with C={C}')

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    # the k-fold split uses index to shuffle the data
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]
    
    # y values come from dataset
    y_train = df_train.churn.values
    y_val = df_val.churn.values
    
    # training and predicting
    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)
    
    # AUC
    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print(f'auc on fold {fold} is {auc}')

print('validation results')
print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

# training the final model
print('training the final model')

dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)

print(f'auc={auc}')

f_out = open(output_file, 'wb')
pickle.dump((dv, model), f_out)
f_out.close()

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')