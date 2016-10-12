import json

import gensim
from gensim.models.doc2vec import TaggedDocument
from sklearn import metrics

# microblog_data_path = "/home/v2john/Dropbox/Personal/Academic/masters/UWaterloo/Academics/ResearchProject/semeval_task/semeval-2017-task-5-subtask-1/Microblog_Trainingdata.json"
microblog_data_path = "/home/darkstar/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/ResearchProject/semeval_task/semeval-2017-task-5-subtask-1/Microblog_Trainingdata.json"

with open(microblog_data_path, "r") as microblog_data_file:
    microblog_data = microblog_data_file.read()

blogpost_list = json.loads(microblog_data)
# print len(blogpost_list)
#print json.dumps(blogpost_list[0]["text"])

all_posts = list()
tag_sentiment_scores = dict()
sentence_count = 0
for sentence in blogpost_list:
    sentences = [sentence["cashtag"]]
    try:
        for span in sentence['spans']:
            sentences.extend(span.split())

        tag = 'SENT_' + str(sentence_count)
        tag_sentiment_scores[tag] = sentence["sentiment score"]
        lsentence = TaggedDocument(words=sentences, tags=[tag])
        print lsentence
        sentence_count += 1
        all_posts.append(lsentence)
    except Exception as e:
        print json.dumps(sentence)
        print e

# print all_posts
model = gensim.models.Doc2Vec(alpha=0.025, min_alpha=0.025, max_vocab_size=None, size=100)
model.build_vocab(all_posts)
model.train(all_posts)

# print model['overbought']
# print model['$TWTR']
# print metrics.pairwise.cosine_similarity(model['overbought'], model['$TWTR'])

print metrics.pairwise.cosine_similarity(model.docvecs['SENT_276'].reshape(1, -1), model.docvecs['SENT_10'].reshape(1, -1))
print tag_sentiment_scores

print model.docvecs.most_similar('SENT_10')
