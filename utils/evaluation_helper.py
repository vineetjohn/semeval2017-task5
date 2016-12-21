import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import log_helper

log = log_helper.get_logger("EvaluationHelper")


def evaluate_task_score(predicted_score_vector, gold_standard_vector):

    # for i in xrange(len(gold_standard_vector)):
    #     log.info("Predicted: " + str(predicted_score_vector[i]) + " : " + str(gold_standard_vector[i]) + ": Actual")

    cosine_smty = \
        cosine_similarity(np.array(predicted_score_vector).reshape(1, -1),
                          np.array(gold_standard_vector).reshape(1, -1))[0][0]

    log.info("Cosine Similarity: " + str(cosine_smty))

    return cosine_smty
