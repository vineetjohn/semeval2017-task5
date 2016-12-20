from sklearn import metrics

from entities.semeval_tagged_line_document import SemevalTaggedLineDocument
from processors.processor import Processor
from utils import doc2vec_helper
from utils import log_helper
from utils import file_helper
from utils import ml_helper
from utils import evaluation_helper

log = log_helper.get_logger("DocvecProcessor")


class DocvecProcessor(Processor):

    def process(self):
        log.info("Began Processing")

        semeval_train_docs = SemevalTaggedLineDocument(self.options.train_headlines_data_path)
        doc2vec_model = doc2vec_helper.init_model(semeval_train_docs)
        log.info("Doc2vec model initialized")

        x_articles, y_train = file_helper.get_article_details(self.options.train_headlines_data_path)

        x_train = list()
        for article in x_articles:
            x_vector = doc2vec_model.infer_vector(article)
            x_train.append(x_vector)

        linear_regression_model = ml_helper.train_linear_model(x_train, y_train)

        x_test_articles, y_true = file_helper.get_article_details(self.options.test_headlines_data_path)

        x_test = list()
        for article in x_test_articles:
            x_vector = doc2vec_model.infer_vector(article)
            x_test.append(x_vector)

        y_pred = linear_regression_model.predict(x_test)

        for i in xrange(len(y_pred)):
            log.info("Predicted: " + str(y_pred[i]) + " - Actual: " + str(y_true[i]))

        log.info("R^2 score: " + str(metrics.r2_score(y_true, y_pred)))
        log.info("Task score: " + str(evaluation_helper.evaluate_task_score(y_pred, y_true)))

        log.info("Completed Processing")
