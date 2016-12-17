from gensim import utils
from gensim.models.doc2vec import TaggedLineDocument, TaggedDocument

from utils import file_helper


class SemevalTaggedLineDocument(TaggedLineDocument):

    def __init__(self, source):
        super(SemevalTaggedLineDocument, self).__init__(source)
        self.semeval_articles = file_helper.get_articles_list(source)
        self.counter = -1

    def __iter__(self):
        for semeval_article in self.semeval_articles:
            self.counter += 1
            yield TaggedDocument(utils.to_unicode(semeval_article['title']).split(), [self.counter])
