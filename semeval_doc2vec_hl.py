import json
from random import shuffle

import gensim
from gensim.models.doc2vec import TaggedDocument

import utils.evaluation_helper as evaluation_helper

headlines_data_path = "/home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/ResearchProject/semeval_task/semeval-2017-task-5-subtask-2/Headline_Trainingdata.json"
# headlines_data_path = "/home/darkstar/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/ResearchProject/semeval_task/semeval-2017-task-5-subtask-2/Headline_Trainingdata.json"

with open(headlines_data_path, "r") as microblog_data_file:
    microblog_data = microblog_data_file.read()

blogpost_list = json.loads(microblog_data)
# print len(blogpost_list)
#print json.dumps(blogpost_list[0]["text"])


all_posts = list()
tag_sentiment_scores = dict()
sentence_count = 0
sum_of_variance = 0
denom = 0

predicted_scores = list()
actual_scores = list()

def verify_sentiment(label):
    global sum_of_variance
    global denom
    global predicted_scores
    global actual_scores
    for (similar_label, cosine_similarity) in model.docvecs.most_similar(label)[0:1]:
        print similar_label + "; similarity: " + str(cosine_similarity) + "; sentiment: " + str(tag_sentiment_scores[similar_label])

        predicted_scores.append(tag_sentiment_scores[similar_label])
        actual_scores.append(tag_sentiment_scores[label])

        diff = abs(float(tag_sentiment_scores[label]) - float(tag_sentiment_scores[similar_label]))
        sum_of_variance += diff
        denom += 2
    # print tag_sentiment_scores[similar_label]

    print "source sentiment score: " + str(tag_sentiment_scores[label])

for sentence in blogpost_list:
    tag = 'SENT_' + str(sentence_count)
    sentences = list()  # [sentence["cashtag"]]
    try:
        sentences.extend(sentence["title"].split())
        tag_sentiment_scores[tag] = sentence["sentiment"]
        lsentence = TaggedDocument(words=sentences, tags=[tag])
        print lsentence
        all_posts.append(lsentence)
    except Exception as e:
        print json.dumps(sentence)
        print e
    sentence_count += 1

# print all_posts
model = gensim.models.Doc2Vec(min_count=1, size=1000, iter=20)
model.build_vocab(all_posts)
model.train(all_posts)

for i in range(100):
    shuffle(all_posts)
    model.train(all_posts)

# verify_sentiment('SENT_1000')

for i in range(1131, 1142):
    label = 'SENT_' + str(i)
    verify_sentiment(label)

print "variance = " + str(sum_of_variance/denom)

# print model.docvecs['SENT_1709']
# print model.docvecs['SENT_1651']

evaluation_helper.evaluate_task_score(predicted_scores, actual_scores)
