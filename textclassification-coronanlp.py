import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

df = pd.read_csv('./Corona_NLP_train.csv', names=['OriginalTweet', 'Sentiment'], encoding='latin-1')

df2 = pd.read_csv('./Corona_NLP_train.csv', names=['OriginalTweet', 'Sentiment'], encoding='latin-1')

# msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})
Xtrain = df.OriginalTweet

ytrain = df.Sentiment

Xtest = df.OriginalTweet

ytest = df.Sentiment

# Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
count_v = CountVectorizer()
Xtrain_dm = count_v.fit_transform(Xtrain)
Xtest_dm = count_v.transform(Xtest)
df = pd.DataFrame(Xtrain_dm.toarray())
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(Xtrain_dm, ytrain)
pred = clf.predict(Xtest_dm)
print('Accuracy Metrics:')
print('Accuracy: ', accuracy_score(ytest, pred))
print('Recall: ', recall_score(ytest, pred, average='macro'))
print('Precision: ', precision_score(ytest, pred, average='macro'))
print('Confusion Matrix: \n', confusion_matrix(ytest, pred))