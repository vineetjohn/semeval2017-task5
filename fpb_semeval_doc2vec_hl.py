import sys
from argparse import ArgumentParser

from processors.fpb_docvec_processor import FPBDocvecProcessor
from utils.options import Options


def main(argv):
    """
    Main function to kick start execution
    :param argv:
    :return: null
    """
    options = parse_args(argv)
    processor = FPBDocvecProcessor(options)
    processor.process()


def parse_args(argv):
    """
    Parses command line arguments form an options object
    :param argv:
    :return:
    """
    parser = ArgumentParser(prog="semeval2015-task5")
    parser.add_argument('--fpb_sentences_file_path', metavar='FPB Sentences File Path',
                        type=str, required=True)
    parser.add_argument('--test_headlines_data_path', metavar='Test Headlines File Path',
                        type=str, required=True)
    parser.add_argument('--docvec_dimension_size', metavar='Dimensions for Doc2Vec',
                        type=int)
    parser.add_argument('--docvec_iteration_count', metavar='Iterations for Doc2Vec',
                        type=int)
    parser.add_argument('--results_file', metavar='File to post results to',
                        type=str)

    return parser.parse_args(argv, namespace=Options)


if __name__ == "__main__":
    main(sys.argv[1:])

