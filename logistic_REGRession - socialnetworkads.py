import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
import seaborn as sns
from sklearn.model_selection import train_test_split
df=pd.read_csv('/kaggle/input/social-network-ads/Social_Network_Ads.csv')
df
X=df.iloc[:,[2,3]].values
Y=df.iloc[:,4].values
print(X)
print(Y)
one_hot_encoded_data = pd.get_dummies(df, columns = (['Gender']))
print(one_hot_encoded_data)
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test= train_test_split(X,Y,test_size = 0.25, random_state=0)

#random_state helps the machine to work with one initialised value by both X and Y.
#change in random_state here changes the prediction value.
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_Train , Y_Train)
Y_Pred = classifier.predict(X_Test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_Test,Y_Pred)
print(cm)
print(accuracy_score(Y_Test, Y_Pred))
r = df['Age']
s = df['EstimatedSalary']
sns.scatterplot(x=r,y=s,hue=df['Purchased'],palette='YlGnBu')
sns.jointplot(x=r,y=s,data=df)
