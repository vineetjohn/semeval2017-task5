import pprint
import re
import json

import numpy as np

from sklearn import linear_model, decomposition, datasets

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import logging
import sys
from time import time


# In[2]:

def train_svm(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = SVC(C=1.0, gamma='auto', kernel='linear')
    svm.fit(X, y)
    return svm


# In[3]:

if __name__ == "__main__":
    with open('/data/ovechtom/ovechtom/2016-Context-opinion/SVM/ambiguous_Adj/Jupyter-svm-test-dataset-ambigAdj.xml') as f:
        test = [tuple(map(str, i.split('<ORIGINAL-REVIEW-SENTENCE>'))) for i in f]
    with open('/data/ovechtom/ovechtom/2016-Context-opinion/SVM/ambiguous_Adj/all-training-Jupyter/Jupyter-svm-training.xml') as f:
        train = [tuple(map(str, i.split('<ORIGINAL-REVIEW-SENTENCE>'))) for i in f]


# In[4]:

# Create the training data class labels
y_train = [d[0] for d in train]
#pprint.pprint(y_train)

# Create the test data class labels
y_test = [t[0] for t in test]
#pprint.pprint(y_test)

# Create the training document corpus list
train_corpus = [d[1] for d in train]

# Create the test document corpus list
test_corpus = [t[1] for t in test]

print("Extracting features from the training data using a sparse vectorizer")
t0 = time()
# Create the TF-IDF vectoriser
vectorizer = TfidfVectorizer(min_df=1, max_features=1000)

#Transform the training corpus
X_train = vectorizer.fit_transform(train_corpus)

duration = time() - t0
#print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_train.shape)
print()

#Transform the test corpus
print("Extracting features from the test data using the same vectorizer")
t0 = time()
X_test = vectorizer.transform(test_corpus)
duration = time() - t0
#print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_test.shape)
print()


# In[5]:

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

pred = classifier.predict(X_test)
#print(accuracy_score(y_test, pred))
print('LogisticRegression score: %f'
  % classifier.fit(X_train, y_train).score(X_test, y_test))
print(confusion_matrix(pred, y_test))


# In[6]:

# Create and train the Support Vector Machine
svm = train_svm(X_train, y_train)


# In[7]:

# Make an array of predictions on the test set
pred = svm.predict(X_test)
pprint.pprint(pred)


# In[8]:

# Output the hit-rate and the confusion matrix for each model
print(svm.score(X_test, y_test))
print(confusion_matrix(pred, y_test))