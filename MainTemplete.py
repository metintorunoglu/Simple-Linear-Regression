# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import the dataset
dataset=pd.read_csv('Data.csv')
X=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 3].values

#missing data
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN', strategy="mean", axis=0)
imp=imputer.fit(X[:, 1:3])
X[:, 1:3]=imp.transform(X[:, 1:3])

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_X=LabelEncoder()
X[:, 0]=le_X.fit_transform(X[:, 0])
enc=OneHotEncoder(categorical_features=[0])
X=enc.fit_transform(X).toarray()
le_y=LabelEncoder()
y=le_y.fit_transform(y)

#Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)









