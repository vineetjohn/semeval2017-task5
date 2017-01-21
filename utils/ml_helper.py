import pandas as pd
import xgboost as xgb

from sklearn import linear_model, svm
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor


def train_linear_regressor(x, y):

    linear_reg_model = linear_model.LinearRegression()
    linear_reg_model.fit(x, y)

    return linear_reg_model


def train_svm_regressor(x, y):

    svm_regressor = svm.LinearSVR()
    svm_regressor.fit(x, y)

    return svm_regressor


def train_gnb_classifier(x, y):

    gnb_classifier = GaussianNB()
    gnb_classifier.fit(x, y)

    return gnb_classifier


def train_xgboost_regressor(x, y):

    xgbooster = XGBRegressor()
    xgbooster.fit(x, y)

    return xgbooster


def persist_model_to_disk(model, model_path):
    joblib.dump(model, model_path)


def get_model_from_disk(model_path):
    return joblib.load(model_path)


def get_confusion_matrix(y_true, y_predicted):

    y_actu = pd.Series(y_true, name='Actual')
    y_pred = pd.Series(y_predicted, name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred)

    return df_confusion
