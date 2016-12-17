from gensim.models import Doc2Vec


def init_model(tagged_articles):

    model = Doc2Vec(min_count=1, size=150, iter=10000, workers=4, window=4, dm=1, hs=1, alpha=1)
    model.build_vocab(tagged_articles)

    model.train(tagged_articles)

    return model
