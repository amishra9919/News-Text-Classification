import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from gensim.models import Word2Vec
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib


data = pd.read_excel("dataset.xlsx")


X = data['Text']
y = data['Category']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


word2vec_model = Word2Vec(sentences=X_train, vector_size=100, window=5, min_count=1, workers=4)
word_vectors = word2vec_model.wv


def word_vectorizer(sentence, model):
    vector = []
    for word in sentence.split():
        if word in model:
            vector.append(model[word])
    if len(vector) == 0:
        return np.zeros(model.vector_size)  
    return np.mean(vector, axis=0)


X_train_vec = [word_vectorizer(sentence, word_vectors) for sentence in X_train]
X_test_vec = [word_vectorizer(sentence, word_vectors) for sentence in X_test]


def evaluate_classifier(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return acc, precision, recall, f1


svm_classifier = SVC()
svm_classifier.fit(X_train_vec, y_train)


svm_scores = evaluate_classifier(svm_classifier, X_test_vec, y_test)
print("Support Vector Machine (SVM) Scores:")
print("Accuracy:", svm_scores[0])
print("Precision:", svm_scores[1])
print("Recall:", svm_scores[2])
print("F1 Score:", svm_scores[3])


joblib.dump(svm_classifier, 'svm_model.pkl')


gnb_classifier = GaussianNB()
gnb_classifier.fit(X_train_vec, y_train)


gnb_scores = evaluate_classifier(gnb_classifier, X_test_vec, y_test)
print("\nGaussian Naive Bayes Scores:")
print("Accuracy:", gnb_scores[0])
print("Precision:", gnb_scores[1])
print("Recall:", gnb_scores[2])
print("F1 Score:", gnb_scores[3])


joblib.dump(gnb_classifier, 'gnb_model.pkl')


bnb_classifier = BernoulliNB()
bnb_classifier.fit(X_train_vec, y_train)


bnb_scores = evaluate_classifier(bnb_classifier, X_test_vec, y_test)
print("\nBernoulli Naive Bayes Scores:")
print("Accuracy:", bnb_scores[0])
print("Precision:", bnb_scores[1])
print("Recall:", bnb_scores[2])
print("F1 Score:", bnb_scores[3])


joblib.dump(bnb_classifier, 'bnb_model.pkl')


rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train_vec, y_train)


rf_scores = evaluate_classifier(rf_classifier, X_test_vec, y_test)
print("\nRandom Forest Scores:")
print("Accuracy:", rf_scores[0])
print("Precision:", rf_scores[1])
print("Recall:", rf_scores[2])
print("F1 Score:", rf_scores[3])


joblib.dump(rf_classifier, 'rf_model.pkl')
