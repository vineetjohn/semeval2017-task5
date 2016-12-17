import json


def get_articles_list(articles_file_path):
    with open(articles_file_path, 'r') as articles_file:
        articles_data = articles_file.read()

    return json.loads(articles_data)


def get_article_details(articles_file_path):

    articles = list()
    sentiment_scores = list()

    semeval_articles = get_articles_list(articles_file_path)

    for semeval_article in semeval_articles:
        sentiment_scores.append(semeval_article['sentiment'])
        articles.append(semeval_article['title'])

    return articles, sentiment_scores
