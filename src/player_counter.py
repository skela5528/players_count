from typing import List

import numpy as np
from PIL import Image


class PlayerCounter:
    def __init__(self, a_descriptor: np.ndarray, b_descriptor: np.ndarray):
        self.a_descriptor = a_descriptor
        self.b_descriptor = b_descriptor

    @staticmethod
    def get_crops_from_boxes(img: np.ndarray, boxes: List[list]):
        crops = []
        for box in boxes:
            left, top, right, bottom = [int(round(val)) for val in box]
            crop = img[top:bottom, left:right, :]
            crops.append(crop)
        return crops

    @staticmethod
    def get_features_from_crops(crops: List[np.ndarray]) -> List[np.ndarray]:
        # mean of each channel
        features = [crop.mean(axis=(0, 1)) for crop in crops]
        return features

    def _classify_box(self):
        ...

    def process_frame(self, img: np.ndarray, boxes: List[list]):
        crops = self.get_crops_from_boxes(img, boxes)
        features = self.get_features_from_crops(crops)

        return features


def debug_player_counter():
    from matplotlib import pyplot as plt

    sample_img_path = '../data/00001.jpg'
    img = np.array(Image.open(sample_img_path))

    boxes = [[1295.2, 682.3, 1442.2, 997.2], [833.4, 406.3, 925.3, 701.0]]
    pc = PlayerCounter(np.array(1), np.array(1))
    crops = pc.get_crops_from_boxes(img, boxes)

    fig, axs = plt.subplots(len(crops))
    for ax, crop in zip(axs, crops):
        ax.imshow(crop)
    plt.show()


if __name__ == '__main__':
    debug_player_counter()
