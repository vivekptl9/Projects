import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from numpy import mean
from numpy import std
import warnings

directory = os.getcwd()
file_name = 'insurance_claims.csv'
file_path = os.path.join(directory, file_name)

# Check if the file exists
if os.path.exists(file_path):
    #print(f"{file_name} found. Opening the file as a pandas DataFrame...")
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    #print(df.head())  # Display the first few rows of the DataFrame
else:
    print(f"{file_name} not found in {directory}.")
df.drop(columns=['_c39'], inplace=True, errors='ignore')
df.replace('?',np.nan,inplace=True)
df['police_report_available'] = df['police_report_available'].fillna(df['police_report_available'].mode()[0])
df['property_damage'] = df['property_damage'].fillna(df['property_damage'].mode()[0])
df['collision_type'] = df['collision_type'].fillna(df['collision_type'].mode()[0])
df['authorities_contacted'] = df['authorities_contacted'].fillna(df['authorities_contacted'].mode()[0])

le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] =le.fit_transform(df[col])
y=df.fraud_reported
feat = list(df.columns)
feat.remove('fraud_reported')
x=df[feat]



X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state =42)
X_train, X_test, y_train, y_test = train_test_split(x,y, stratify=y, random_state =1)

smt = SMOTE()
X_train,y_train = smt.fit_resample(X_train,y_train)
np.bincount(y_train)


def get_stacking():
    # define the base models
    level0 = list()
    level0.append(('DT', DecisionTreeClassifier(max_depth=10, random_state=5)))
    level0.append(('RF', RandomForestClassifier(n_estimators=500)))
    level0.append(('KNN', KNeighborsClassifier(5)))
    level0.append(('XGB', XGBClassifier(objective= 'binary:logistic', use_label_encoder=False)))
    level1 = LogisticRegression() 
    # define the stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=10)
    return model

def get_models():
    models = dict()
    models['DT'] = DecisionTreeClassifier(max_depth=10)
    models['RF'] = RandomForestClassifier(n_estimators=500)
    models['KNN'] = KNeighborsClassifier(5)
    #models['ADA'] = AdaBoostClassifier(n_estimators=500)
    models['XGB'] = XGBClassifier(objective= 'binary:logistic', eval_metric='logloss')
    models['Stacking'] = get_stacking()
    return models

warnings.filterwarnings("ignore", category=UserWarning)
models = get_models()

# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=5)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores

# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model, X_train, y_train)
    results.append(scores)
    names.append(name)
    print('>%s %.2f (%.2f)' % (name, mean(scores), std(scores)))