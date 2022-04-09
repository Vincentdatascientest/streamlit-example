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


# STRUCTURE STREAMLITE

st.title('Projet CaPYtal')



# DF
url = 'https://drive.google.com/file/d/1WnBL1zEQo1KVEinJBN6ZfyeW4U4d7Yk3/view?usp=sharing'
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]

df = pd.read_csv(path, sep = ';')
df = df.rename(columns={'y': 'deposit'})
df['deposit']=df['y']
df.drop(["y"], axis = 1, inplace = True) 



#PREPROCESSING

df_num = df.select_dtypes(include=['int64', 'float64']).columns
scaler = preprocessing.StandardScaler().fit(df[df_num])
df[df_num] = pd.DataFrame(scaler.transform(df[df_num]))


df_cat = df.select_dtypes(include=['object']).columns

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for feat in df_cat:
    df[feat] = le.fit_transform(df[feat].astype(str))


target = df['deposit']
feats = df.drop('deposit',axis=1)

X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.2, random_state=12)

st.dataframe(df)


#OVERSAMPLING SMOTE

smote = SMOTE(random_state = 101)
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)

print('After OverSampling, the shape of X_train: {}'.format(X_train_over.shape)) 
print('After OverSampling, the shape of y_train: {} \n'.format(y_train_over.shape)) 
  
print("After OverSampling, counts of label '1': {}".format(sum(y_train_over == 1))) 
print("After OverSampling, counts of label '0': {}".format(sum(y_train_over == 0))) 



#ENTRAINEMENT DU MODELE

st.title('Entrainement du modèle')



options = ['Random Forest', 'KNN', 'Decision Tree']
modele_choisi = st.selectbox(label='Choix de modèle', options=options)

RFC = ensemble.RandomForestClassifier(n_jobs=-1, random_state=321, max_features='auto', n_estimators=700)
KNN = KNeighborsClassifier(metric='manhattan', n_neighbors=1)
dtree = DecisionTreeClassifier(criterion = 'gini', max_depth=9, min_samples_leaf=1, min_samples_split=2)

@st.cache 
def train_model(modele_choisi): 
    if modele_choisi == options[0]:
        model = ensemble.RandomForestClassifier(n_jobs=-1, random_state=321, max_features='auto', n_estimators=700)
    elif modele_choisi == options[1]:
        model = KNeighborsClassifier(metric='manhattan', n_neighbors=1)
    else :
        model = DecisionTreeClassifier(criterion = 'gini', max_depth=9, min_samples_leaf=1, min_samples_split=2)
        
    model.fit(X_train_over, y_train_over)
    score = model.score(X_test, y_test)
    return score



st.write('Score Test', train_model(modele_choisi))
