import json
from sklearn import svm
import numpy as np
from random import shuffle

import gensim
from gensim.models.doc2vec import TaggedDocument

headlines_data_path = "/home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/ResearchProject/semeval_task/semeval-2017-task-5-subtask-2/Headline_Trainingdata.json"
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
model = gensim.models.Doc2Vec(alpha=0.025, min_alpha=0.025, max_vocab_size=None, size=100, iter=20)
model.build_vocab(all_posts)
model.train(all_posts)

# for i in range(5):
#     shuffle(all_posts)
#     model.train(all_posts)

# print model.docvecs["SENT_1"]
# print tag_sentiment_scores["SENT_1"]

x_docvecs = list()
y_scores = list()
for i in xrange(len(model.docvecs)):
    tag = "SENT_" + str(i)
    x_docvecs.append(model.docvecs[tag])
    y_scores.append(int( (tag_sentiment_scores[tag] * 10) + 10))

# print [y-10 for y in y_scores]

svm_classifier = svm.LinearSVC(C=5.0)
svm_classifier.fit(X=x_docvecs, y=y_scores)

for i in xrange(30):
    print (svm_classifier.predict(np.array(x_docvecs[i]).reshape(1, -1))[0] - 10, y_scores[i] - 10)
