import json

from utils import log_helper

log = log_helper.get_logger("FileHelper")


def get_articles_list(articles_file_path):
    with open(articles_file_path, 'r') as articles_file:
        articles_data = articles_file.read()

    return json.loads(articles_data)


def get_article_details(articles_file_path):

    articles = list()
    sentiment_scores = list()

    semeval_articles = get_articles_list(articles_file_path)

    for semeval_article in semeval_articles:
        if "sentiment" in semeval_article.keys():
            sentiment_scores.append(semeval_article['sentiment'])
        articles.append(semeval_article['title'])

    return articles, sentiment_scores


def annotate_test_set(test_headlines_data_path, y_test):

    with open(test_headlines_data_path, 'r') as test_headlines_file:
        articles = json.load(test_headlines_file)

        for i in range(len(y_test)):
            articles[i]["sentiment"] = y_test[i]

    with open(test_headlines_data_path, 'w') as test_headlines_file:
        json.dump(articles, test_headlines_file, indent=4, separators=(',', ': '))

    log.info(json.dumps(articles))

    return None
