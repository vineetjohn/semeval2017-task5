import json

microblog_data_path = "/home/v2john/PycharmProjects/SemevalDoc2Vec/semeval_doc2vec/semeval-2017-task-5-subtask-1/Microblog_Trialdata-full.json"

with open(microblog_data_path, "r") as microblog_data_file:
    microblog_data = microblog_data_file.read()

blogpost_list = json.loads(microblog_data)

print blogpost_list[0]
