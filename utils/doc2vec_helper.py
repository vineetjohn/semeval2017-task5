from gensim.models import Doc2Vec


def init_model(tagged_articles):

    model = Doc2Vec(min_count=1, size=800, iter=40, workers=1, window=4, seed=1)
    model.build_vocab(tagged_articles)

    model.train(tagged_articles)

    return model
