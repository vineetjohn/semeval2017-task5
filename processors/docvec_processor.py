from processors.processor import Processor
from utils import log_helper

log = log_helper.get_logger("AmazonReviewProcessor")


class DocvecProcessor(Processor):

    def process(self):
        log.info("Began Processing")

        log.info("Completed Processing")
