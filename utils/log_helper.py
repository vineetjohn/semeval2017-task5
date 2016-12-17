import logging, sys


def get_logger(logger_name):

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format='[%(asctime)s]: %(name)s : %(levelname)s : %(message)s'
    )
    application_logger = logging.getLogger(logger_name)

    return application_logger
