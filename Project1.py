# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 13:37:09 2022

@author: biatt
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


df= pd.read_csv(r"C:\Users\biatt\Desktop\cancer_prediction.csv")
df.info()
df.isnull()

## Representation of the percentage of Male and Female as per the datafeild --

df.GENDER.nunique()
df.GENDER.count()
df.GENDER.value_counts()

plt.pie(df.GENDER.value_counts(),labels=("male","female"),autopct="%1.1f%%",colors=("Blue","hotpink"))
plt.legend(title="GENDER")
plt.show()



## Percentage of Smokers and Non Smokers -- 

df.SMOKING.nunique()







df.SMOKING.count()
df.SMOKING.value_counts()

plt.pie(df.SMOKING.value_counts(),labels=("Smokers","Non-Smokers"),colors=("m","c"),autopct="%1.2d%%")
plt.legend(title="% of Smokers")
plt.show()


df
df.loc[(df.SMOKING==1)&(df.LUNG_CANCER=="YES")] ## Doesn't Smokes but has Lcancer 
df.loc[(df.SMOKING==2)&(df.LUNG_CANCER=="YES")] ## Smokes and has Lcancer
df.loc[(df.SMOKING==1)&(df.LUNG_CANCER=="NO")] ## Doesn't Smokes and dont have
df.loc[(df.SMOKING==2)&(df.LUNG_CANCER=="NO")] ## Smokes but doesn't




df.PEER_PRESSURE.value_counts()

plt.pie(df.PEER_PRESSURE.value_counts(),labels=("Yes","No"),colors=("orange","yellow"),autopct="%1.0d%%")
plt.legend(title="Peer-pressure")
plt.show()

df.AGE.nunique()

df.AGE.value_counts()









import sklearn 
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
df["LUNG_CANCER"]=label_encoder.fit_transform(df["LUNG_CANCER"])
df["LUNG_CANCER"].unique()

## NOW AFTER LABEL ENCODING  YES=1 AND NO=0

df.LUNG_CANCER

df.info()
df.drop(["WHEEZING","YELLOW_FINGERS"],axis="columns",inplace=True)
df.drop("ALLERGY",inplace=True)

c=df.copy()
c.drop("GENDER",axis="columns",inplace=True)

x=c.iloc[:,:-1]
y=c.iloc[:,-1]


## Training and Testing Part ---

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1500)

x_train
x_test
y_train
y_test



## Algorithms -- (Classificastion) --


## Support Vector Machine --

from sklearn.svm import SVC
sr =SVC()

sr =SVC(kernel = "rbf",C =50,gamma=10)
sr =SVC(kernel = "sigmoid")

sr.fit(x_train,y_train)
sr.score(x_test,y_test)
sr.score(x_train,y_train)
 


## Decision Tree --

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion ="gini")
dt = DecisionTreeClassifier(criterion ="gini",min_samples_split=11)
dt.fit(x_train,y_train)
dt.score(x_test,y_test)
dt.score(x_train,y_train)
AA= dt.predict(x_test)
AA
y_test

dt = DecisionTreeClassifier(criterion ="entropy")
dt.fit(x_train,y_train)
dt.score(x_test,y_test)
dt.score(x_train,y_train)
dt.predict(x_test)
y_test


## Random Forest -- 

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=2000)
rf = RandomForestClassifier(n_estimators=1000,oob_score = True)
rf.fit(x_train,y_train)
# rf.score(x_test,y_test)
print(rf.oob_score_)


from sklearn.metrics import accuracy_score
y_pred = rf.predict(x_test)
accuracy_score(y_test,y_pred)


## Naive Bayes -- 

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(x_train,y_train)
mnb.score(x_test,y_test)
mnb.score(x_train,y_train)

from sklearn.naive_bayes import GaussianNB
mnb = GaussianNB()
mnb.fit(x_train,y_train)
mnb.score(x_test,y_test)
mnb.score(x_train,y_train)


from sklearn.naive_bayes import BernoulliNB
mnb = BernoulliNB()
mnb.fit(x_train,y_train)
mnb.score(x_test,y_test)
mnb.score(x_train,y_train)


#Logistic Regression --

from sklearn.linear_model import LogisticRegression
modelLR=LogisticRegression()
modelLR.fit(x_train,y_train)
modelLR.score(x_test,y_test)
modelLR.score(x_train,y_train)








































