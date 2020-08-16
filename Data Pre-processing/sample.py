# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset=pd.read_csv(r'path_to_dataset/Data Pre-processing/Data.csv')
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:,3].values

#Converting DataFrame from Object
df1 = pd.DataFrame(X)
df2 = pd.DataFrame(Y)

#Taking care of missing data
#axis = 0 -> row
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean',axis=0)
#fit -> to calculate mean
imputer = imputer.fit(X[:,1:3])
#transform the calculated value back to original dataset
X[:,1:3] = imputer.transform(X[:,1:3])

#Encoding the Categorial Data
#Giving Labels and then converting to OneHotEncoding
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

#Splitting the Dataset into Test set and Training Set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)