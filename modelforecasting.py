import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle

df=pd.read_csv("https://raw.githubusercontent.com/kruphacm/mini-project/main/forecasting%20data.csv")
df['N OR AN'] = df['N OR AN'].map({'N':0.0,'AN':1.0})
X=df[['HEART BEAT','HEART BEAT']]
Y=df['N OR AN']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
knn = KNeighborsClassifier(n_neighbors=4)
  
knn.fit(X,Y)

pickle.dump(knn, open('modelheartbeat.pkl','wb'))

model = pickle.load(open('modelheartbeat.pkl','rb'))

X1=df[['BLOOD OXYGEN LEVEL','BLOOD OXYGEN LEVEL']]
Y1=df['N OR AN']
knn1 = KNeighborsClassifier(n_neighbors=4)
  
knn1.fit(X1,Y1)
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.33)
knn1.fit(X1_test,Y1_test)

pickle.dump(knn1, open('modelbloodoxygen.pkl','wb'))

model = pickle.load(open('modelbloodoxygen.pkl','rb'))

df['BLOOD PRESSURE']=df['BLOOD PRESSURE'].str.split("/")
df['SYSTOLIC']=df['BLOOD PRESSURE'].str[0]
df['DIASTOLIC']=df['BLOOD PRESSURE'].str[1]
X2,Y2,X3,Y3=df[['SYSTOLIC','SYSTOLIC']],df['N OR AN'],df[['DIASTOLIC','DIASTOLIC']],df['N OR AN']
knn2 = KNeighborsClassifier(n_neighbors=4)
  
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size=0.33)
knn2.fit(X2,Y2)

pickle.dump(knn2, open('modelsystolic.pkl','wb'))

model = pickle.load(open('modelsystolic.pkl','rb'))

knn3 = KNeighborsClassifier(n_neighbors=4)
X3_train, X3_test, Y3_train, Y3_test = train_test_split(X3, Y3, test_size=0.33)
knn3.fit(X3,Y3)
pickle.dump(knn3, open('modeldiastolic.pkl','wb'))

model = pickle.load(open('modeldiastolic.pkl','rb'))

X4=df[['TEMPERATURE','TEMPERATURE']]
Y4=df['N OR AN']
knn4 = KNeighborsClassifier(n_neighbors=4)
  
knn4.fit(X4,Y4)
X4_train, X4_test, Y4_train, Y4_test = train_test_split(X4, Y4, test_size=0.33)

pickle.dump(knn4, open('modeltemperature.pkl','wb'))

model = pickle.load(open('modeltemperature.pkl','rb'))