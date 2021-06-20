import enum
from typing import List

import numpy as np
from PIL import Image


class TeamClass(enum.IntEnum):
    unknown = 0
    team_a = 1
    team_b = 2
    referee = 10


class PlayerCounter:
    def __init__(self, a_descriptor: np.ndarray, b_descriptor: np.ndarray, referee_descriptor: np.ndarray, distance_thresh: float):
        self.a_descriptor = a_descriptor
        self.b_descriptor = b_descriptor
        self.referee_descriptor = referee_descriptor

        self.distance_thresh = distance_thresh

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

    @staticmethod
    def _get_distances_to_descriptor(features: np.ndarray, descriptor: np.ndarray):
        square_errors = (features - descriptor) ** 2
        mse = square_errors.mean(axis=1)
        rmse = mse ** 0.5
        return rmse

    def count_players(self, features: List[np.ndarray]):
        features = np.array(features)
        dist_a = self._get_distances_to_descriptor(features, self.a_descriptor)
        dist_b = self._get_distances_to_descriptor(features, self.b_descriptor)
        dist_ref = self._get_distances_to_descriptor(features, self.referee_descriptor)

        team_predictions = np.zeros(len(features))  # 0-unknown, 1-a, 2-b, 10-referee
        for ind in range(len(features)):
            da = dist_a[ind]
            db = dist_b[ind]

            # unknown
            if min(da, db) > self.distance_thresh:
                continue

            # team a
            if da < db:
                team_predictions[ind] = TeamClass.team_a
            # team b
            else:
                team_predictions[ind] = TeamClass.team_b

        # referee
        ref_candidate_ind = np.argmin(dist_ref)
        if dist_ref[ref_candidate_ind] < self.distance_thresh:
            team_predictions[ref_candidate_ind] = TeamClass.referee.value
        return team_predictions

    def process_frame(self, img: np.ndarray, boxes: List[list]):
        crops = self.get_crops_from_boxes(img, boxes)
        features = self.get_features_from_crops(crops)
        team_predictions = self.count_players(features)
        return team_predictions


def calculate_team_descriptor(same_team_features: List[np.ndarray]) -> np.ndarray:
    features = np.array(same_team_features)
    team_descriptor = np.round(features.mean(axis=0))
    return team_descriptor


def debug_player_counter():
    from matplotlib import pyplot as plt

    sample_img_path = '../data/00001.jpg'
    img = np.array(Image.open(sample_img_path))

    boxes = [[1295.2, 682.3, 1442.2, 997.2], [833.4, 406.3, 925.3, 701.0]]
    pc = PlayerCounter(np.array(1), np.array(1), np.array(1), 1)
    crops = pc.get_crops_from_boxes(img, boxes)

    fig, axs = plt.subplots(len(crops))
    for ax, crop in zip(axs, crops):
        ax.imshow(crop)
    plt.show()


if __name__ == '__main__':
    debug_player_counter()
