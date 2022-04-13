# PACKAGES

import pandas as pd
import numpy as np
import streamlit as st

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection

from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn import tree

import shap
from IPython.display import display
import matplotlib.pyplot as plt

import pickle



# STRUCTURE STREAMLITE

st.title('Projet CaPYtal')
st.markdown('A travers ce projet, nous allons pouvoir prédire si un individu est susceptible de souscrire à la proposition ')



# DF
df = pd.read_csv('bank-additional-full.csv', sep = ';')
df['deposit']=df['y']
df.drop(["y"], axis = 1, inplace = True) 

st.dataframe(df.head(4))



#SIDEBAR

st.sidebar.subheader('Variable du prospect')

age = st.sidebar.slider('Age', 18, 95, value=20, step=1)
duration = st.sidebar.slider('Duration', 0, 5000, value=2000, step=1)
campaign = st.sidebar.slider('Campaign', 1, 6, value=2, step=1)
pdays = st.sidebar.slider('pdays', 0, 999, value=3, step=1)
previous = st.sidebar.slider('previous', 0, 7, value=0, step=1)
emp_var_rate = st.sidebar.slider('Emp.var.rate', -3.0, 2.0, value=-0.1, step= 0.1)
cons_price_idx = st.sidebar.slider('Cons.price.idx', 90.000,95.000, value=93.200, step=0.001)
cons_conf_idx = st.sidebar.slider('Cons.conf.idx', -48.0, -33.0, value=-42.0, step=0.1)
euribor3m = st.sidebar.slider('Euribor3m', 0.800, 5.000, value=1.25, step=0.001)
nr_employed = st.sidebar.slider ('nr.employed', 5000.0, 5300.0, value=5228.0, step=0.1)


#PREPROCESSING

df_num = df.select_dtypes(include=['int64', 'float64']).columns
scaler = preprocessing.StandardScaler().fit(df[df_num])
df[df_num] = pd.DataFrame(scaler.transform(df[df_num]))


df_cat = df.select_dtypes(include=['object']).columns


le = LabelEncoder()
for feat in df_cat:
    df[feat] = le.fit_transform(df[feat].astype(str))


target = df['deposit']
feats = df.drop('deposit',axis=1)

X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.2, random_state=12)




#OVERSAMPLING SMOTE

smote = SMOTE(random_state = 101)
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)

print('After OverSampling, the shape of X_train: {}'.format(X_train_over.shape)) 
print('After OverSampling, the shape of y_train: {} \n'.format(y_train_over.shape)) 
  
print("After OverSampling, counts of label '1': {}".format(sum(y_train_over == 1))) 
print("After OverSampling, counts of label '0': {}".format(sum(y_train_over == 0))) 



#ENTRAINEMENT DU MODELE

st.subheader('Entrainement du modèle')



options = ['Random Forest', 'KNN', 'Decision Tree']
modele_choisi = st.selectbox(label='Choix de modèle', options=options)

RFC = ensemble.RandomForestClassifier(n_jobs=-1, random_state=321, max_features='auto', n_estimators=700)
KNN = KNeighborsClassifier(metric='manhattan', n_neighbors=1)
dtree = DecisionTreeClassifier(criterion = 'gini', max_depth=9, min_samples_leaf=1, min_samples_split=2)


def train_model(modele_choisi): 
    if modele_choisi == options[0]:
        model = ensemble.RandomForestClassifier(n_jobs=-1, random_state=321, max_features='auto', n_estimators=700)
    elif modele_choisi == options[1]:
        model = KNeighborsClassifier(metric='manhattan', n_neighbors=1)
    else :
        model = DecisionTreeClassifier(criterion = 'gini', max_depth=9, min_samples_leaf=1, min_samples_split=2)
        
    model.fit(X_train_over, y_train_over)
    score = model.score(X_test, y_test)
    
    prediction = model.predict(X_test)
    crosstab = pd.crosstab(y_test, prediction, rownames=['Classe réelle'], colnames=['Classe prédite'])
    return score, crosstab



st.write('Score Test', train_model(modele_choisi)


st.sidebar.subheader('Variable du prospect')

age = st.sidebar.slider('Age', 18, 95, value=20, step=1)
duration = st.sidebar.slider('Duration', 0, 5000, value=2000, step=1)
campaign = st.sidebar.slider('Campaign', 1, 6, value=2, step=1)
pdays = st.sidebar.slider('pdays', 0, 999, value=3, step=1)
previous = st.sidebar.slider('previous', 0, 7, value=0, step=1)
emp_var_rate = st.sidebar.slider('Emp.var.rate', -3.0, 2.0, value=-0.1, step= 0.1)
cons_price_idx = st.sidebar.slider('Cons.price.idx', 90.000,95.000, value=93.200, step=0.001)
cons_conf_idx = st.sidebar.slider('Cons.conf.idx', -48.0, -33.0, value=-42.0, step=0.1)
euribor3m = st.sidebar.slider('Euribor3m', 0.800, 5.000, value=1.25, step=0.001)
nr_employed = st.sidebar.slider ('nr.employed', 5000.0, 5300.0, value=5228.0, step=0.1)




input_list = [age, duration, campaign, pdays, previous, emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nr_employed]
input_preds_class = model_choisi.predict(input_list)
input_preds_proba = model_choisi.predict_proba(input_list)



df_input = pd.DataFrame([input_list], columns=['age', 'housing', 'loan', 'contact', 'date', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'emp_var_rate', 'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed']


#INTERPRETATION













