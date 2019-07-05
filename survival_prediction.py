import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

data = pd.read_csv("titanic_train.csv")
test_data = pd.read_csv('test.csv')
#data.head()
#data.shape
data.describe()

data.isnull().sum()

data.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)

data.drop(['Cabin'],axis=1,inplace=True)

def impute_age(l):
    age=l[0]
    pclass=l[1]
    
    if pd.isnull(age):
        if pclass==1:
            return 38
        elif pclass==2:
            return 29
        else:
            return 25
    else:
        return age
