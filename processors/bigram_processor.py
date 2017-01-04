from sklearn import model_selection, linear_model
from sklearn.feature_extraction.text import CountVectorizer

from processors.processor import Processor
from utils import file_helper
from utils import log_helper

log = log_helper.get_logger("BigramProcessor")


class BigramProcessor(Processor):

    def process(self):
        log.info("Began Processing")

        x_train_articles, y_train = file_helper.get_article_details(self.options.train_headlines_data_path)
        x_test_articles, y_test = file_helper.get_article_details(self.options.test_headlines_data_path)

        log.info("Extracting articles and scores")
        x_train_articles.extend(x_test_articles)
        y_train.extend(y_test)

        log.info("Vectorizing articles")
        vectorizer = CountVectorizer(ngram_range=(1,10))
        x_vectors = vectorizer.fit_transform(x_train_articles)

        log.info("Testing Prediction model")
        scores = model_selection.cross_val_score(linear_model.LinearRegression(), x_vectors,
                                                 y_train, cv=10, scoring='r2')

        log.info(scores)
        log.info("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        log.info("Completed Processing")
