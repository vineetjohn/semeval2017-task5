#!/usr/bin/env bash
for dimension_size in 50 51
do
    for iteration_count in 20 21
    do
        export PYTHONPATH=/home/vineet/semeval2017-task5 && \
        /usr/bin/python2.7 /home/v2john/PycharmProjects/semeval2017-task5/semeval_doc2vec_hl.py \
        --train_headlines_data_path /home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/ResearchProject/semeval_task/semeval-2017-task-5-subtask-2/Headline_Trainingdata.json \
        --test_headlines_data_path /home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/ResearchProject/semeval_task/semeval-2017-task-5-subtask-2/Headline_Trialdata.json \
        --results_file /home/v2john/Dropbox/Personal/Academic/Masters/UWaterloo/ResearchProject/semeval_task/semeval-2017-task-5-subtask-2/results_`/bin/date +%Y%m%d`.txt \
        --docvec_dimension_size ${dimension_size} \
        --docvec_iteration_count ${iteration_count}
    done
done
