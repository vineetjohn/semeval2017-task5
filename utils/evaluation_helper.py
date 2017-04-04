import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils import log_helper

log = log_helper.get_logger("EvaluationHelper")


def evaluate_task_score(y_true, y_pred):

    cosine_smty = \
        cosine_similarity(np.array(y_pred).reshape(1, -1),
                          np.array(y_true).reshape(1, -1))[0][0]

    log.info("Cosine Similarity: " + str(cosine_smty))

    return cosine_smty


