from argparse import ArgumentParser

import sys

from processors.bigram_processor import BigramProcessor
from processors.docvec_processor import DocvecProcessor
from processors.docvec_processor_crossval import DocvecProcessorCrossval
from utils.options import Options


def main(argv):
    """
    Main function to kick start execution
    :param argv:
    :return: null
    """
    options = parse_args(argv)
    processor = BigramProcessor(options)
    processor.process()


def parse_args(argv):
    """
    Parses command line arguments form an options object
    :param argv:
    :return:
    """
    parser = ArgumentParser(prog="semeval2015-task5")
    parser.add_argument('--train_headlines_data_path', metavar='Training Headlines File Path',
                        type=str, required=True)
    parser.add_argument('--test_headlines_data_path', metavar='Test Headlines File Path',
                        type=str, required=True)
    parser.add_argument('--docvec_dimension_size', metavar='Dimensions for Doc2Vec',
                        type=int, required=False)
    parser.add_argument('--docvec_iteration_count', metavar='Iterations for Doc2Vec',
                        type=int, required=False)
    parser.add_argument('--results_file', metavar='File to post results to',
                        type=str, required=True)
    parser.add_argument('--validate', metavar='Mode: validate',
                        type=bool, required=False)
    parser.add_argument('--annotate', metavar='Mode: annotate',
                        type=str, required=False)
    parser.add_argument('--max_ngram', metavar='Max N-Gram',
                        type=int, required=False)
    parser.add_argument('--min_ngram', metavar='Min N-Gram',
                        type=int, required=False)

    return parser.parse_args(argv, namespace=Options)


if __name__ == "__main__":
    main(sys.argv[1:])
