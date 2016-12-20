from argparse import ArgumentParser

import sys

from processors.docvec_processor import DocvecProcessor
from utils.options import Options


def main(argv):
    """
    Main function to kick start execution
    :param argv:
    :return: null
    """
    options = parse_args(argv)
    processor = DocvecProcessor(options)
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
    parser.add_argument('--min_dimension_size', metavar='Minimun Dimensions for Doc2Vec',
                        type=int)
    parser.add_argument('--max_dimension_size', metavar='Maximum Dimensions for Doc2Vec',
                        type=int)
    parser.add_argument('--min_docvec_iter', metavar='Minimun Iterations for Doc2Vec',
                        type=int)
    parser.add_argument('--max_docvec_iter', metavar='Maximum Dimensions for Doc2Vec',
                        type=int)
    parser.add_argument('--results_file', metavar='File to post results to',
                        type=str, required=True)

    return parser.parse_args(argv, namespace=Options)


if __name__ == "__main__":
    main(sys.argv[1:])

