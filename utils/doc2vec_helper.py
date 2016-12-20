from gensim.models import Doc2Vec


def init_model(tagged_articles, dimension_size, iterations):

    model = Doc2Vec(min_count=1, size=dimension_size, iter=iterations, workers=1, window=4, seed=1)
    model.build_vocab(tagged_articles)

    model.train(tagged_articles)

    return model
