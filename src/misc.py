import logging
import os
from collections import namedtuple

# Box = namedtuple('Box', 'left top right bottom')


def get_logger(log_path: str = None, level=logging.DEBUG) -> logging.Logger:
    global LOGGER
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_format = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    handlers = []
    if log_path is not None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(log_format)
        handlers.append(file_handler)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(log_format)
    handlers.append(stdout_handler)
    logging.basicConfig(level=level, handlers=handlers, datefmt='%Y-%m-%d %H:%M:%S')
    LOGGER = logging.getLogger()
    return LOGGER


LOGGER = get_logger()
