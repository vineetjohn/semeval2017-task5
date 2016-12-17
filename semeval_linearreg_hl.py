import json
from random import shuffle

import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn import linear_model

import utils.evaluation_helper as evaluation_helper

# input file in the semeval format
# headlines_data_path = "/home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/ResearchProject/semeval_task/semeval-2017-task-5-subtask-2/combined.json"
headlines_data_path = \
    "/home/darkstar/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/ResearchProject" + \
    "/semeval_task/semeval-2017-task-5-subtask-2/Headline_Trainingdata.json"


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

# arranging input and output vectors for regression model training
x_docvecs = list()
y_scores = list()
for i in xrange(len(model.docvecs) - 15):
    tag = "SENT_" + str(i)
    x_docvecs.append(model.docvecs[tag])
    y_scores.append(tag_sentiment_scores[tag])

print "Training the prediction model"
# training the svm regression model
# svm_classifier = svm.LinearSVR()
# svm_classifier.fit(X=x_docvecs, y=y_scores)

# training the linear regression model
regr = linear_model.LinearRegression()
regr.fit(X=x_docvecs, y=y_scores)
print "Prediction model trained successfully"

predicted_scores = list()
actual_scores = list()

# making the predictions
for i in xrange(len(model.docvecs) - 15, len(model.docvecs)):
    tag = "SENT_" + str(i)
    predicted_score = round(regr.predict(np.array(model.docvecs[tag]).reshape(1, -1))[0], 2)
    actual_score = tag_sentiment_scores[tag]

    predicted_scores.append(predicted_score)
    actual_scores.append(actual_score)

    print blogpost_list[i]
    print (predicted_score, actual_score, abs(round((predicted_score - actual_score), 2)))

# evaluating the accuracy
evaluation_helper.evaluate_task_score(predicted_scores, actual_scores)
