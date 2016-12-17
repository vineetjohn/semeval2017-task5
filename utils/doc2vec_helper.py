from gensim.models import Doc2Vec


def init_model(tagged_articles):

    model = Doc2Vec(min_count=1, size=200, iter=500, workers=2, window=4, dm=0)
    model.build_vocab(tagged_articles)

    model.train(tagged_articles)

    return model
