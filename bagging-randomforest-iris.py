import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import plot_tree

iris=load_iris()
X, y =iris.data, iris.target
classifier = RandomForestClassifier(n_estimators = 10)

classifier.fit(X,y)

bag_classifier = BaggingClassifier(estimator = classifier, n_estimators =12)
bag_classifier.fit(X,y)

plt.figure(figsize =(12, 10))
plot_tree(bag_classifier.estimator_[0],filled = True,fontsize = 11)
feature_names = iris.feature_names
class_names = list(iris.target_names)
plt.title("Decision Tree in Random Forest")
plt.show()
