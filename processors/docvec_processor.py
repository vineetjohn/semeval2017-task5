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

        semeval_train_docs = SemevalTaggedLineDocument(self.options.train_headlines_data_path)

        results_file = open(self.options.results_file, 'w')

        for dimension_size in xrange(self.options.min_dimension_size, self.options.max_dimension_size + 1):
            for iteration_count in xrange(self.options.min_docvec_iter, self.options.max_docvec_iter + 1):
                doc2vec_model = doc2vec_helper.init_model(semeval_train_docs, dimension_size, iteration_count)
                log.info("Doc2vec model initialized with " + str(dimension_size) +
                         "dimensions and " + str(iteration_count) + "iterations")

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

                test_result_dict = dict()
                test_result_dict['dimension_size'] = dimension_size
                test_result_dict['iteration_count'] = iteration_count
                test_result_dict['r2_score'] = metrics.r2_score(y_true, y_pred)
                test_result_dict['semeval_score'] = evaluation_helper.evaluate_task_score(y_pred, y_true)

                results_file.write(str(json.dumps(test_result_dict)) + "\n")

        results_file.close()
        log.info("Completed Processing")
