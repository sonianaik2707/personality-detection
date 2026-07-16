#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer
import sys
sys.path.append('../src')
from preprocess import load_and_prepare, split_data
df = load_and_prepare('mbti_1.csv')
X_train, X_test, y_train, y_test = split_data(df)
vectorizer = TfidfVectorizer(
max_features=5000, # keep top 5000 words
ngram_range=(1, 2), # unigrams + bigrams
stop_words='english'
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test) # transform only, NOT fit
print('Train shape:', X_train_tfidf.shape)
print('Test shape: ', X_test_tfidf.shape)


# In[ ]:




