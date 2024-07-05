import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

datasets=pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
X=datasets.iloc[:,[0,1]].values
Y=datasets.iloc[:,8].values

from sklearn.model_selection import train_test_split
X_Train,X_Test,Y_Train,Y_Test=train_test_split(X,Y,test_size=0.1,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_Train=sc_X.fit_transform(X_Train)
X_Test=sc_X.transform(X_Test)

from sklearn.svm import SVC
classifier=SVC(kernel='rbf')
classifier.fit(X_Train, Y_Train)
Y_Pred= classifier.predict(X_Test)

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
print('Accuracy: %.3f'%accuracy_score(Y_Pred,Y_Test))
print("Precision:%.3f "%precision_score(Y_Pred, Y_Test))
print("F1 score: %.3f"%f1_score(Y_Pred, Y_Test))
print("Recall score: %.3f"%recall_score(Y_Pred, Y_Test))

from matplotlib.colors import ListedColormap
X_Set, Y_Set = X_Test, Y_Test
X1, X2 = np.meshgrid(np.arange(start = X_Set[:, 0].min() - 1, stop = X_Set[:,
0].max() + 1, step = 0.01),
np.arange(start = X_Set[:, 1].min() - 1, stop = X_Set[:, 1].max() +
1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),
X2.ravel()]).T).reshape(X1.shape),alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_Set)):
 plt.scatter(X_Set[Y_Set == j, 0], X_Set[Y_Set == j, 1],
 c = ListedColormap(('white', 'blue'))(i), label = j)
plt.title("Support Vector Machine(Test Set)")
plt.legend()
plt.show()