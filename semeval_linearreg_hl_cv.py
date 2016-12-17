import json
from random import shuffle

import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn import linear_model
from sklearn.model_selection import KFold

import utils.evaluation_helper as evaluation_helper

# input file in the semeval format
headlines_data_path = "/home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/ResearchProject/semeval_task/semeval-2017-task-5-subtask-2/combined.json"
# headlines_data_path = "/home/darkstar/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/ResearchProject/semeval_task/semeval-2017-task-5-subtask-2/Headline_Trainingdata.json"


# reading the file contents into a data dictionary format
with open(headlines_data_path, "r") as microblog_data_file:
    microblog_data = microblog_data_file.read()

blogpost_list = json.loads(microblog_data)

all_posts = list()
tag_sentiment_scores = dict()
sentence_count = 0
sum_of_variance = 0
shuffle_count = 0

# This is to convert the headlines into the input data type for doc2vec
for sentence in blogpost_list:
    tag = 'SENT_' + str(sentence_count)
    sentences = list()
    try:
        sentences.extend(sentence["title"].split())
        tag_sentiment_scores[tag] = sentence["sentiment"]
        lsentence = TaggedDocument(words=sentences, tags=[tag])
        all_posts.append(lsentence)
    except Exception as e:
        print json.dumps(sentence)
        print e
    sentence_count += 1


# model training
print "Training the doc2vec model ..."
model = Doc2Vec(min_count=2, size=1000, iter=50, dm=0)
model.build_vocab(all_posts)
model.train(all_posts)

# shuffling and re-training the model
for i in range(shuffle_count):
    shuffle(all_posts)
    print "Shuffles left: " + str(shuffle_count - i)
    model.train(all_posts)
print "Doc2Vec model trained successfully"

print len(model.docvecs)
y_values = np.asarray(tag_sentiment_scores.values())
print len(y_values)

# using tenfold cross-validation

kf = KFold(n_splits=10)
kf.get_n_splits(model.docvecs)

for train_index, test_index in kf.split(model.docvecs):

    X_train, X_test = model.docvecs[train_index], model.docvecs[test_index]
    y_train, y_test = y_values[train_index], y_values[test_index]

    print "Training the prediction model"

    # training the svm regression model
    # svm_classifier = svm.LinearSVR()
    # svm_classifier.fit(X=x_docvecs, y=y_scores)

    # training the linear regression model
    regr = linear_model.LinearRegression()
    regr.fit(X=X_train, y=y_train)
    print "Prediction model trained successfully"

    # making the predictions

    y_predict = regr.predict(X=X_test)

    # evaluating the accuracy
    evaluation_helper.evaluate_task_score(y_predict, y_test)
