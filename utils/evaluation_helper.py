import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import log_helper

log = log_helper.get_logger("EvaluationHelper")


def evaluate_task_score(predicted_score_vector, gold_standard_vector):

    log.info("predicted_score_vector: " + str(predicted_score_vector))
    log.info("gold_standard_vector: " + str(gold_standard_vector))

    predicted_score_norm = np.linalg.norm(predicted_score_vector)
    gold_standard_norm = np.linalg.norm(gold_standard_vector)
    log.info("predicted_score_norm: " + str(predicted_score_norm))
    log.info("gold_standard_norm: " + str(gold_standard_norm))

    cosine_weight = np.divide(predicted_score_norm, gold_standard_norm)
    cosine_smty = \
        cosine_similarity(np.array(predicted_score_vector).reshape(1, -1),
                          np.array(gold_standard_vector).reshape(1, -1))[0][0]

    log.info("cosine_weight " + str(cosine_weight))
    log.info("cosine_similarity " + str(cosine_smty))

    final_cosine_score = cosine_weight * cosine_smty
    log.info("final_cosine_score " + str(final_cosine_score))

    return final_cosine_score
