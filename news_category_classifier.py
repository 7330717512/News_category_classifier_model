import numpy as np
import pandas as pd

df = pd.read_csv('news-data.csv')

df.head()

df.tail()

df.shape

df.category.value_counts()

df.info()

df.isnull().sum()

df = df.drop_duplicates()
df.shape

df.tail()

df = df.reset_index()

df.drop('index',inplace=True,axis=1)

df.tail()



import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

corpus = []
for i in range(0, len(df)):
  text = re.sub('[^a-zA-Z]', ' ', df['text'][i])
  text = text.lower()
  text = ' '.join((word) for word in text.split() if word not in stopwords.words('english'))
  corpus.append(text)

"""Tf-Idf Vectorizer"""

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=2000,ngram_range=(1,2))

X = tfidf.fit_transform(corpus).toarray()

X[4:9]

X.shape

y = df.category

y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)

X_train.shape , X_test.shape

"""Model Building"""

#LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
lr = LogisticRegression()

lr.fit(X_train,y_train)
y_pred1 = lr.predict(X_test)

print("******** LogisticRegression ********")
print(f'Accuracy is : {accuracy_score(y_pred1,y_test)}')

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train,y_train)
y_pred3 = mnb.predict(X_test)


print("******** MultinomialNB ********")
print(f'Accuracy is : {accuracy_score(y_pred3,y_test)}')

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

rf.fit(X_train,y_train)
y_pred4 = rf.predict(X_test)

print("******** RandomForestClassifier ********")
print(f'Accuracy is : {accuracy_score(y_pred4,y_test)}')

from xgboost import XGBClassifier
xgb = XGBClassifier()

xgb.fit(X_train,y_train)
y_pred2 = xgb.predict(X_test)

print("******** XGBClassifier ********")
print(f'Accuracy is : {accuracy_score(y_pred2,y_test)}')

"""Predict on Random text"""

pred = mnb.predict(tfidf.transform(['NEW DELHI: Prime Minister Narendra Modi on Friday choked with emotion while thanking doctors, frontline workers during video conference with them. "As a servant of Kashi, I thank everyone in Varanasi, especially the doctors, nurses, technicians, ward boys and ambulance drivers who have done a commendable work," PM said.']))

print(pred)

pred = mnb.predict(tfidf.transform(["India’s largest public sector bank State Bank of India (SBI) on Friday reported a standalone net profit of Rs 6,450.75 crore for quarter ended March 2021 (Q4FY21) aided by fewer provisions on bad loans. The lender’s PAT was 80.14 per cent higher than previous year’s profit of Rs 3,580.8 crore. On a quarterly basis, the bottom line expanded 24.14 per cent."]))

print(pred)

import pickle
f = open('mnb.pickle', 'wb')
pickle.dump(mnb, f)
f.close()

f = open('tfidf.pickle', 'wb')
pickle.dump(tfidf, f)
f.close()