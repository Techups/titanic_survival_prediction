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

data['Age']=data[['Age','Pclass']].apply(impute_age,axis=1)

g=pd.get_dummies(data['Sex'],drop_first=True)
e=pd.get_dummies(data['Embarked'],drop_first=True)

data=pd.concat([data,g,e],axis=1)

data.drop(['Sex','Embarked'],axis=1,inplace=True)

data.drop('Fare',axis=1,inplace=True)

sns.heatmap(data.corr(),cmap='coolwarm')

X = data.drop("Survived", axis=1)
y = data["Survived"]

test_data.drop(['Cabin','Ticket','Name'],axis=1,inplace=True)

z = pd.get_dummies(test_data['Sex'],drop_first=True)
v = pd.get_dummies(test_data['Embarked'],drop_first=True)

test_data = pd.concat([test_data,z,v],axis=1)

test_data.drop(['Sex','Embarked'],axis=1,inplace=True)

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
	
test_data['Age'] = test_data[['Age','Pclass']].apply(impute_age,axis=1)

test_data.drop('Fare',axis=1,inplace=True)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
confu_mat = confusion_matrix(y_test,y_pred)
print(confu_mat)

print(classification_report(y_pred,y_test))

ids = test_data['PassengerId']
prediction = logreg.predict(test_data.drop('PassengerId',axis=1))

output = pd.DataFrame({'PassengerId':ids,'Survived':prediction})
output.to_csv('titanic_submission_one.csv',index=False)
    

