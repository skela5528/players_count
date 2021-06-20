import argparse
import json
import os
from configparser import ConfigParser
from typing import List, Union

import cv2
import numpy as np
from PIL import Image

from config_parser import get_config
from detector import Detector, ModelName
from misc import LOGGER, draw_boxes
from player_counter import PlayerCounter, TeamClass


class Runner:
    config: ConfigParser

    @classmethod
    def get_detector(cls) -> Detector:
        LOGGER.info(f'Init Detector ...')
        # parse detector config
        model_name_string = cls.config.get('Detector', 'model_name')
        assert hasattr(ModelName, model_name_string)
        model_name = ModelName[model_name_string]
        score_thresh = cls.config.getfloat('Detector', 'score_thresh')
        nms_thresh = cls.config.getfloat('Detector', 'nms_thresh')
        width = cls.config.getint('General', 'input_img_width')
        height = cls.config.getint('General', 'input_img_height')
        device = cls.config.get('Detector', 'device')

        # create Detector object
        detector = Detector(model_name, score_thresh, nms_thresh, (width, height), device)
        return detector

    @classmethod
    def get_player_counter(cls) -> PlayerCounter:
        LOGGER.info(f'Init PlayerCounter ...')

        # parse counter config
        a_descriptor = json.loads(cls.config.get('PlayersCounter', 'team_a_descriptor'))
        b_descriptor = json.loads(cls.config.get('PlayersCounter', 'team_b_descriptor'))
        a_descriptor, b_descriptor = np.array(a_descriptor), np.array(b_descriptor)
        ref_descriptor = json.loads(cls.config.get('PlayersCounter', 'referee_descriptor'))
        ref_descriptor = np.array(ref_descriptor)
        distance_thresh = cls.config.getint('PlayersCounter', 'distance_thresh')

        # create PlayerCounter object
        counter = PlayerCounter(a_descriptor, b_descriptor, ref_descriptor, distance_thresh)
        return counter

    @classmethod
    def img_stream_generator(cls):
        input_frames_dir = cls.config.get('General', 'input_frames_dir')
        assert os.path.exists(input_frames_dir)

        names = sorted(os.listdir(input_frames_dir))
        for name in names:
            img_path = os.path.join(input_frames_dir, name)
            img = Image.open(img_path)
            yield img, name

    @staticmethod
    def save_json(data: Union[dict, list], out_dir: str):
        LOGGER.info(f'Saving json: {out_dir}')
        data_path = os.path.join(out_dir, 'debug.json')
        with open(data_path, 'w') as f:
            json.dump(data, f)

    @staticmethod
    def _add_data_to_json(data: list, name: str, features: List[np.ndarray]):
        # add line to debug_json
        line = dict()
        line['name'] = name
        line['features'] = {i: ftr.astype(int).tolist() for i, ftr in enumerate(features)}
        data.append(line)

    @classmethod
    def main(cls):
        detector = cls.get_detector()
        counter = cls.get_player_counter()
        out_dir = cls.config.get('General', 'output_dir')
        os.makedirs(out_dir, exist_ok=True)

        debug_json = []
        for img, name in cls.img_stream_generator():
            LOGGER.info(f'process img: {name}')
            img_np = np.array(img)
            boxes, scores = detector.get_person_boxes(img)
            team_predictions, features = counter.process_frame(img_np, boxes)

            team_a_count = sum([1 for x in team_predictions if x == TeamClass.team_a.value])
            team_b_count = sum([1 for x in team_predictions if x == TeamClass.team_b.value])
            referee = sum([1 for x in team_predictions if x == TeamClass.referee.value])
            count_str = f'team a: {team_a_count} | team b: {team_b_count} | referee: {referee}'

            # draw
            texts = [f'{i}: {int(s * 100)}' for i, s in enumerate(scores)]
            draw = draw_boxes(img, boxes, texts=texts, classes=team_predictions)
            draw = cv2.putText(draw, count_str, (100, 50),  cv2.FONT_HERSHEY_DUPLEX, 2, (255,  140, 0), 2)

            # save draw
            cv2.imwrite(os.path.join(out_dir, name), cv2.cvtColor(draw, cv2.COLOR_RGB2BGR))

            # add line to debug_json
            cls._add_data_to_json(debug_json, name, features)
        cls.save_json(debug_json, out_dir)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', default='../config.ini')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    Runner.config = get_config(args.config_path)
    Runner.main()
