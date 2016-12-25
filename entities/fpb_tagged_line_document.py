from gensim import utils
from gensim.models.doc2vec import TaggedLineDocument, TaggedDocument


class FPBTaggedLineDocument(TaggedLineDocument):

    def __init__(self, source):
        super(FPBTaggedLineDocument, self).__init__(source)
        self.source = source

    def __iter__(self):
        with utils.smart_open(self.source) as fin:
            for item_no, line in enumerate(fin):
                text = line.rsplit(None, 1)[0]
                yield TaggedDocument(text.split(), [item_no])

    def get_label_list(self):
        global num_label
        label_list = list()
        with open(self.source) as source:
            for line in source:
                label = line.rsplit(None, 1)[-1]
                if label == '.@neutral':
                    num_label = 0
                elif label == '.@positive':
                    num_label = 1
                elif label == '.@negative':
                    num_label = -1
                label_list.append(num_label)
        return label_list

    def get_phrases(self):
        with open(self.source) as source:
            for line in source:
                text = line.rsplit(None, 1)[0]
                yield text
