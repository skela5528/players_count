import logging
import os
from typing import Optional, List

import cv2
import numpy as np
from PIL import Image


def get_logger(log_path: str = None, level=logging.INFO) -> logging.Logger:
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


COLORS = {0: (255, 255, 255),
          1: (250, 0, 0),
          2: (0, 0, 250),
          10: (0, 0, 0)}


def draw_boxes(img: Image, boxes: list, texts: Optional[List[str]] = None, color: tuple = (0, 0, 250),
               classes: Optional[list] = None) -> np.ndarray:
    img = np.array(img)
    for ind, box in enumerate(boxes):
        left, top, right, bottom = [int(round(val)) for val in box]
        if classes is not None:
            color = COLORS.get(classes[ind])
        cv2.rectangle(img, (left, top), (right, bottom), color, 2)

        if texts is not None:
            text_org = (left + 12, top + 32)
            cv2.putText(img, texts[ind], text_org, cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
    return img
