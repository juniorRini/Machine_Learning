import pandas as pd

dataframe = pd.read_csv('/kaggle/input/titanic-dataset/Titanic-Dataset.csv')
# print(dataframe)

dataframe = dataframe.dropna()

X = dataframe[['Pclass', 'Age', 'SibSp', 'Parch']].values
Y = dataframe['Survived'].values
from sklearn.model_selection import train_test_split

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.6, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_Train = sc.fit_transform(X_Train)
X_Test = sc.transform(X_Test)

# from sklearn.svm import SVC
# svc=SVC(probability=True, kernel='linear')

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import AdaBoostClassifier

accuracy_1 = []
estimators = [10, 20, 30, 40, 50.60, 70, 80, 90, 100]
for i in range(10, 100, 10):
    Classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=(1)), n_estimators=i, learning_rate=1,
                                    random_state=0)  # , base_estimator=svc)

    Classifier.fit(X_Train, Y_Train)
    Y_Pred = Classifier.predict(X_Test)
    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(Y_Test, Y_Pred)
    print("Accuracy = ", accuracy)

    from sklearn.metrics import precision_score

    print("Precision = ", precision_score(Y_Test, Y_Pred))
    accuracy_1.append(accuracy)
import matplotlib.pyplot as plt

plt.plot(estimators, accuracy_1)
plt.xlabel("No. of estimators")
plt.ylabel("Accuracy prediction")

import seaborn as sns

plt.figure(figsize=(30, 30))
for i, s in enumerate(Classifier.estimators_):
    if i < 60:
        plt.subplot(5, 12, i + 1)
        plot_tree(s, filled=True, fontsize=(14), feature_names=['Pclass', 'Age', 'SibSp', 'Parch'],
                  class_names=['Not Survived', 'Survived'])
        plt.title(f'S {i + 1}')

plt.tight_layout()
plt.show()
