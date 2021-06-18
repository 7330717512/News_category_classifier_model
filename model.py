# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

df =pd.read_csv('news-data.csv')
y = df['category']

X_train, X_test, y_train, y_test = train_test_split(
                                             df['text'], y, 
                                             test_size=0.33, 
                                             random_state=53)
count_vectorizer = CountVectorizer(stop_words='english')

count_train = count_vectorizer.fit_transform(X_train.values)
count_test = count_vectorizer.transform(X_test.values)
nb_classifier = MultinomialNB()
nb_classifier.fit(count_train, y_train)

f = open('nb_classifier.pickle', 'wb')
pickle.dump(nb_classifier, f)
f.close()

f = open('count_vectorizer.pickle', 'wb')
pickle.dump(count_vectorizer, f)
f.close()


pred = nb_classifier.predict(count_test)
print(metrics.accuracy_score(y_test, pred))






