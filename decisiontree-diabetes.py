import numpy as np
import pandas as pd

df=pd.read_csv("/kaggle/input/diabetescsv/diabetes.csv")
print(df)
from sklearn.model_selection import train_test_split
X=df.iloc[:,:8].values
Y=df.iloc[:,-1].values
print(X)
print(Y)
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y,test_size = 0.25, random_state=43)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state=40)
classifier.fit(X_Train , Y_Train)
Y_Pred = classifier.predict(X_Test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm=confusion_matrix(Y_Test,Y_Pred)
print(cm)
print(accuracy_score(Y_Test, Y_Pred))