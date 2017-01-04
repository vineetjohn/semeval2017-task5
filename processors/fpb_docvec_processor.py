from sklearn import metrics

from entities.fpb_tagged_line_document import FPBTaggedLineDocument
from processors.processor import Processor
from utils import doc2vec_helper
from utils import evaluation_helper
from utils import file_helper
from utils import log_helper
from utils import ml_helper

log = log_helper.get_logger("FPBDocvecProcessor")


class FPBDocvecProcessor(Processor):

    def process(self):
        log.info("Began Processing")

        fpb_training_docs = FPBTaggedLineDocument(self.options.fpb_sentences_file_path)

        doc2vec_model = \
            doc2vec_helper.init_model(
                fpb_training_docs, self.options.docvec_dimension_size, self.options.docvec_iteration_count
            )
        log.info("Doc2vec model initialized with " + str(self.options.docvec_dimension_size) +
                 " dimensions and " + str(self.options.docvec_iteration_count) + " iterations")
        label_list = fpb_training_docs.get_label_list()

        log.info("Re-training document vectors")
        x_train = list()
        for i in xrange(len(label_list)):
            x_vector = doc2vec_model.infer_vector(fpb_training_docs.get_phrases())
            x_train.append(x_vector)

        log.info("Training ML model")
        linear_regression_model = ml_helper.train_linear_regressor(x_train, label_list)

        log.info("Predicting test set")
        x_test_articles, y_true = file_helper.get_article_details(self.options.test_headlines_data_path)
        x_test = list()
        for article in x_test_articles:
            x_vector = doc2vec_model.infer_vector(article)
            x_test.append(x_vector)
        y_pred = linear_regression_model.predict(x_test)

        test_result_dict = dict()
        test_result_dict['dimension_size'] = self.options.docvec_dimension_size
        test_result_dict['iteration_count'] = self.options.docvec_iteration_count
        test_result_dict['r2_score'] = metrics.r2_score(y_true, y_pred)
        test_result_dict['semeval_score'] = evaluation_helper.evaluate_task_score(y_pred, y_true)

        log.info("Test result: " + str(test_result_dict))

        log.info("Completed Processing")
