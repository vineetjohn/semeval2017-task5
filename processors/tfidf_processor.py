from sklearn import model_selection, linear_model, svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import xgboost as xgb
from xgboost import XGBRegressor

from processors.processor import Processor
from utils import file_helper
from utils import log_helper
from utils.ml_helper import train_xgboost_regressor

log = log_helper.get_logger("TFIDFProcessor")


class TFIDFProcessor(Processor):

    def process(self):
        log.info("Began Processing")

        if self.options.validate:
            x_train_articles, y_train = file_helper.get_article_details(self.options.train_headlines_data_path)
            x_test_articles, y_test = file_helper.get_article_details(self.options.test_headlines_data_path)

            log.info("Extracting articles and scores")
            x_train_articles.extend(x_test_articles)
            y_train.extend(y_test)

            vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
            x_vectors = vectorizer.fit_transform(x_train_articles)

            log.info("Testing Prediction model")
            scores = model_selection.cross_val_score(XGBRegressor(), x_vectors,
                                                     y_train, cv=10, scoring='r2')
            mean_score = scores.mean()
            log.info("Accuracy: %0.2f (+/- %0.2f)" % (mean_score, scores.std() * 2))

        elif self.options.annotate:
            raise RuntimeError("Run mode not implemented")
        else:
            raise RuntimeError("Invalid run mode. Valid modes are 'validate' and 'annotate'")

        log.info("Completed Processing")
