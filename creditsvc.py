import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
a=pd.read_csv("C:/Users/Lokesh/Desktop/cdac ai material/machine learning/lab/creditcard.csv")
print(a.head())
a.drop(('Time'),axis=1,inplace=True)
a.drop(('Amount'),axis=1,inplace=True)
print(a.info())
x=a.drop('Class',axis=1)
y=a['Class']
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=100)
print('train of x_train is:',x_train.shape)
print('train of y_train is:',y_train.shape)
print('train of x_test is:',x_test.shape)
print('train of y_test is:',y_test.shape)
from sklearn.svm import SVC
model=SVC()
print(model)
print(model)
model.fit(x_train,y_train)
predicteddata=model.predict(x_test)
print("predicteddata",predicteddata)
print("y_test",y_test)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,predicteddata)
print("Accuracy of model is:",accuracy)
from sklearn.metrics import classification_report
cr=classification_report(y_test,predicteddata)
print(cr)












