from sklearn import model_selection, linear_model
from sklearn.feature_extraction.text import CountVectorizer

from processors.processor import Processor
from utils import file_helper
from utils import log_helper

log = log_helper.get_logger("BigramProcessor")
min_ngram_range = range(1, 11)
max_ngram_range = range(1, 11)


class BigramProcessor(Processor):

    def process(self):
        log.info("Began Processing")

        x_train_articles, y_train = file_helper.get_article_details(self.options.train_headlines_data_path)
        x_test_articles, y_test = file_helper.get_article_details(self.options.test_headlines_data_path)

        log.info("Extracting articles and scores")
        x_train_articles.extend(x_test_articles)
        y_train.extend(y_test)

        max_score = 0
        best_config_tuple = (0, 0)

        for x in min_ngram_range:
            for y in max_ngram_range:
                try:
                    log.info("Vectorizing articles")
                    vectorizer = CountVectorizer(ngram_range=(x,y))
                    x_vectors = vectorizer.fit_transform(x_train_articles)

                    log.info("Testing Prediction model")
                    scores = model_selection.cross_val_score(linear_model.LinearRegression(), x_vectors,
                                                             y_train, cv=10, scoring='r2')

                    mean_score = scores.mean()
                    if mean_score > max_score:
                        max_score = mean_score
                        best_config_tuple = (x, y)

                    log.info("Accuracy: %0.2f (+/- %0.2f)" % (mean_score, scores.std() * 2))
                except ValueError:
                    log.error("Value error for ngram config (" + str(x) + ", " + str(y) + ")")

        log.info("Best score is " + str(max_score))
        log.info("Best ngrams(min, max): " + str(best_config_tuple))

        log.info("Completed Processing")
