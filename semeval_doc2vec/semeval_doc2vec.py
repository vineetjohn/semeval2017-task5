import json
import gensim


microblog_data_path = "/home/v2john/Dropbox/Personal/Academic/masters/UWaterloo/Academics/ResearchProject/semeval_task/semeval-2017-task-5-subtask-1/Microblog_Trainingdata.json"

with open(microblog_data_path, "r") as microblog_data_file:
    microblog_data = microblog_data_file.read()

blogpost_list = json.loads(microblog_data)
print len(blogpost_list)
#print json.dumps(blogpost_list[0]["text"])

all_posts = list()
for sentence in blogpost_list:
    # print sentence["text"]
    try:
        for span in sentence['spans']:
            all_posts.append(span)
    except Exception as e:
        print json.dumps(sentence)


# print all_posts
model = gensim.models.Doc2Vec(alpha=0.025, min_alpha=0.025, size=1000)

model.build_vocab(all_posts)
