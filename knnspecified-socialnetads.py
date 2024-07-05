
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score

dataframe = pd.read_csv("/kaggle/input/social-network-ads/Social_Network_Ads.csv")
# Create dummy variables for Gender
sex = pd.get_dummies(dataframe['Gender'])

# Concatenate dummy variables and drop Gender column
dataframe = pd.concat([dataframe, sex], axis=1)
dataframe = dataframe.drop(['Gender'], axis=1)

# Select features and target
x = dataframe[['Age', 'EstimatedSalary']].values
y = dataframe['Purchased'].values

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

# Feature scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Initialize KNN classifier with Euclidean distance metric
knn_classifier = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

# Fit the classifier to the training data
knn_classifier.fit(x_train, y_train)

# Predict the target values on the test set
y_pred = knn_classifier.predict(x_test)

# Calculate accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)

# Lists to store accuracy values
accuracy_values = []

# Experiment with different values of n_neighbors
for n_neighbors in range(1, 21):  # Try values from 1 to 20
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean')
    knn_classifier.fit(x_train, y_train)
    y_pred = knn_classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_values.append(accuracy)  # Store accuracy value

    print(f"n_neighbors = {n_neighbors}")
    print("Accuracy:", accuracy)
    print("=" * 30)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), accuracy_values, marker='o')
plt.title('Accuracy vs. Number of Neighbors')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(range(1, 21))
plt.grid(True)
plt.show()

