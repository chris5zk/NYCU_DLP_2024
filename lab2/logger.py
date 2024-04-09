import os
import logging

os.makedirs('./log', exist_ok=True)

def get_logger():

    logger = logging.getLogger('resnet50')
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler('./log/resnet50.log')
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

logger = get_logger()
