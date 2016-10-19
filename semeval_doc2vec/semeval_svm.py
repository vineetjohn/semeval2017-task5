import json
from random import shuffle

import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn import svm

import utils.evaluation_helper as evaluation_helper

headlines_data_path = "/home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/ResearchProject/semeval_task/semeval-2017-task-5-subtask-2/combined.json"
# headlines_data_path = "/home/darkstar/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/ResearchProject/semeval_task/semeval-2017-task-5-subtask-2/Headline_Trainingdata.json"

with open(headlines_data_path, "r") as microblog_data_file:
    microblog_data = microblog_data_file.read()

blogpost_list = json.loads(microblog_data)

all_posts = list()
tag_sentiment_scores = dict()
sentence_count = 0
sum_of_variance = 0


for sentence in blogpost_list:
    tag = 'SENT_' + str(sentence_count)
    sentences = list()  # [sentence["cashtag"]]
    try:
        sentences.extend(sentence["title"].split())
        tag_sentiment_scores[tag] = sentence["sentiment"]
        lsentence = TaggedDocument(words=sentences, tags=[tag])
        # print lsentence
        all_posts.append(lsentence)
    except Exception as e:
        print json.dumps(sentence)
        print e
    sentence_count += 1

# print all_posts
model = Doc2Vec(min_count=1, size=1000, iter=20, dm=0)
model.build_vocab(all_posts)
model.train(all_posts)

for i in range(100):
    shuffle(all_posts)
    model.train(all_posts)

x_docvecs = list()
y_scores = list()
for i in xrange(len(model.docvecs) - 15):
    tag = "SENT_" + str(i)
    x_docvecs.append(model.docvecs[tag])
    y_scores.append(tag_sentiment_scores[tag])

# print [y-10 for y in y_scores]

svm_classifier = svm.LinearSVR()
svm_classifier.fit(X=x_docvecs, y=y_scores)

predicted_scores = list()
actual_scores = list()

for i in xrange(len(y_scores) - 15, len(y_scores)):
    predicted_score = round(svm_classifier.predict(np.array(x_docvecs[i]).reshape(1, -1))[0], 2)
    actual_score = y_scores[i]

    predicted_scores.append(predicted_score)
    actual_scores.append(actual_score)

    print blogpost_list[i]
    print (predicted_score, actual_score, abs(round((predicted_score - y_scores[i]), 2)))

evaluation_helper.evaluate_task_score(predicted_scores, actual_scores)

# print len(y_scores)
