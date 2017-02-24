import json

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

        if self.options.validate:
            semeval_train_docs = SemevalTaggedLineDocument(self.options.train_headlines_data_path)

            doc2vec_model = \
                doc2vec_helper.init_model(
                    semeval_train_docs, self.options.docvec_dimension_size, self.options.docvec_iteration_count
                )
            log.info("Doc2vec model initialized with " + str(self.options.docvec_dimension_size) +
                     " dimensions and " + str(self.options.docvec_iteration_count) + " iterations")

            x_articles, y_train = file_helper.get_article_details(self.options.train_headlines_data_path)

            x_train = list()
            for article in x_articles:
                x_vector = doc2vec_model.infer_vector(article)
                x_train.append(x_vector)

            linear_regression_model = ml_helper.train_linear_regressor(x_train, y_train)

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

            log.info(test_result_dict)

            # with open(self.options.results_file, 'a') as results_file:
            #     results_file.write(str(json.dumps(test_result_dict)) + "\n")

        elif self.options.annotate:
            semeval_train_docs = SemevalTaggedLineDocument(self.options.train_headlines_data_path)

            doc2vec_model = \
                doc2vec_helper.init_model(
                    semeval_train_docs, self.options.docvec_dimension_size, self.options.docvec_iteration_count
                )
            log.info("Doc2vec model initialized with " + str(self.options.docvec_dimension_size) +
                     " dimensions and " + str(self.options.docvec_iteration_count) + " iterations")

            x_articles, y_train = file_helper.get_article_details(self.options.train_headlines_data_path)

            x_train = list()
            for article in x_articles:
                x_vector = doc2vec_model.infer_vector(article)
                x_train.append(x_vector)

            linear_regression_model = ml_helper.train_linear_regressor(x_train, y_train)

            x_test_articles, y_true = file_helper.get_article_details(self.options.test_headlines_data_path)

            x_test = list()
            for article in x_test_articles:
                x_vector = doc2vec_model.infer_vector(article)
                x_test.append(x_vector)

            y_pred = linear_regression_model.predict(x_test)

            log.info("Annotating test set")
            file_helper.annotate_test_set(self.options.test_headlines_data_path, y_pred)

        else:
            raise RuntimeError("Invalid run mode. Valid modes are 'validate' and 'annotate'")

        log.info("Completed Processing")
