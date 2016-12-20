import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import log_helper

log = log_helper.get_logger("EvaluationHelper")


def evaluate_task_score(predicted_score_vector, gold_standard_vector):

    log.debug("predicted_score_vector: " + str(predicted_score_vector))
    log.debug("gold_standard_vector: " + str(gold_standard_vector))

    cosine_smty = \
        cosine_similarity(np.array(predicted_score_vector).reshape(1, -1),
                          np.array(gold_standard_vector).reshape(1, -1))[0][0]

    return cosine_smty
