import json

import gensim
from gensim.models.doc2vec import TaggedDocument
from sklearn import metrics

headlines_data_path = "/home/v2john/Dropbox/Personal/Academic/masters/UWaterloo/Academics/ResearchProject/semeval_task/semeval-2017-task-5-subtask-2/Headline_Trainingdata.json"

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

def verify_sentiment(label):
    global sum_of_variance
    global denom
    for (similar_label, cosine_similarity) in model.docvecs.most_similar(label)[0:1]:
        print similar_label + "; similarity: " + str(cosine_similarity) + "; sentiment: " + str(tag_sentiment_scores[similar_label])
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
model = gensim.models.Doc2Vec(alpha=0.025, min_alpha=0.025, max_vocab_size=None, size=1000, iter=20)
model.build_vocab(all_posts)
model.train(all_posts)

# verify_sentiment('SENT_1000')

for i in range(1131, 1142):
    verify_sentiment('SENT_' + str(i))

print "variance = " + str(sum_of_variance/denom)

# print model.docvecs['SENT_1709']
# print model.docvecs['SENT_1651']
