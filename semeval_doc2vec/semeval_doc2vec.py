import json
import gensim


microblog_data_path = "/home/darkstar/Dropbox/Personal/Academic/Masters/UWaterloo/Academics/ResearchProject/semeval_task/semeval-2017-task-5-subtask-1/Microblog_Trialdata-full.json"

with open(microblog_data_path, "r") as microblog_data_file:
    microblog_data = microblog_data_file.read()

blogpost_list = json.loads(microblog_data)
print len(blogpost_list)
#print json.dumps(blogpost_list[0]["text"])

all_posts = list()
for sentence in blogpost_list:
    # print sentence["text"]
    try:
        all_posts.append(sentence["text"])
        continue
    except Exception as e:
        print json.dumps(sentence)

    try:
        all_posts.append(sentence["message"]["body"])
    except Exception as e:
        print json.dumps(sentence)


print all_posts




model = gensim.models.Doc2Vec(documents=all_posts, size=300)