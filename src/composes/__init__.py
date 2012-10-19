import logging

class NullHandler(logging.Handler):
    """For python versions <= 2.6; same as `logging.NullHandler` in 2.7."""
    def emit(self, record):
        pass

logger = logging.getLogger(__name__)
if len(logger.handlers) == 0:    # To ensure reload() doesn't add another one
    logger.addHandler(NullHandler())
    
#logging.basicConfig(filename='composes.log', filemode='w+',level=logging.DEBUG, format = "")
