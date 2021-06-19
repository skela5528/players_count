import enum
from typing import Optional, List

import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large


class ModelName(enum.IntEnum):
    ssd_lite = 1


def to_numpy(values: torch.Tensor) -> np.ndarray:
    return values.detach().cpu().numpy()


class Detector:
    def __init__(self, model_name: ModelName, score_thresh: float, nms_thresh: float, input_wh: tuple, device: str = 'cuda'):
        self.input_wh = input_wh
        self.device = device
        self.net_dim: int = 320

        self.model = self._load_model(model_name, score_thresh, nms_thresh, device)
        self.preprocess = self._get_img_transforms(input_wh[1], self.net_dim)

    @staticmethod
    def _get_img_transforms(input_height: int, net_in_size: int) -> transforms.Compose:
        img_transforms = transforms.Compose([
            transforms.CenterCrop(input_height),
            transforms.Resize(size=net_in_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
        return img_transforms

    @staticmethod
    def _load_model(model_name: ModelName, score_thresh: float, nms_thresh: float, device: str) -> torch.nn.Module:
        model = None
        if model_name == ModelName.ssd_lite:
            model = ssdlite320_mobilenet_v3_large(pretrained=True, pretrained_backbone=True, score_thresh=score_thresh, nms_thresh=nms_thresh)
        elif ...:
            ...
        else:
            raise ValueError
        model.to(device)
        model.eval()
        return model

    @staticmethod
    def _filter_predictions(predictions: dict, white_list_labels: list = [1]) -> Optional[dict]:
        indexes_keep = [i for i, label in enumerate(predictions['labels']) if label in white_list_labels]
        indexes_keep = torch.tensor(indexes_keep)
        result = None
        if len(indexes_keep):
            result = {key: values[indexes_keep] for key, values in predictions.items()}
        return result

    @staticmethod
    def _resize_boxes(boxes: np.ndarray, origin_wh: tuple, net_dim: int) -> List[list]:
        # resize center crop
        width, height = origin_wh
        x_shift = (width - height) / 2
        scale_factor = height / net_dim

        # resize x
        boxes[:, 0] = boxes[:, 0] * scale_factor + x_shift
        boxes[:, 2] = boxes[:, 2] * scale_factor + x_shift

        # resize y
        boxes[:, 1] = boxes[:, 1] * scale_factor
        boxes[:, 3] = boxes[:, 3] * scale_factor

        # to list of lists
        boxes = boxes.tolist()  # type: List[List[float, float, float, float]]
        return boxes

    def get_person_boxes(self, img: Image) -> (List[list], List[float]):
        input_img = self.preprocess(img)
        input_img = input_img.unsqueeze(0)
        input_img = input_img.to(self.device)

        predictions = self.model(input_img)
        predictions = self._filter_predictions(predictions[0])
        boxes, scores = predictions['boxes'], predictions['scores']
        boxes, scores = to_numpy(boxes), to_numpy(scores)
        boxes = self._resize_boxes(boxes, self.input_wh, self.net_dim)
        return boxes, scores


def draw_boxes(img: Image, boxes: list, color: tuple = (20, 20, 180)) -> np.ndarray:
    img = np.array(img)
    for box in boxes:
        left, top, right, bottom = [int(round(val)) for val in box]
        cv2.rectangle(img, (left, top), (right, bottom), color, 2)
    return img


if __name__ == '__main__':
    p = '/home/cortica/Documents/my/git_personal/data/ml6/frames_png/00001.png'
    img_ = Image.open(p)

    det = Detector(ModelName.ssd_lite, .15, .5, (1920, 1080))
    boxes_, scores_ = det.get_person_boxes(img_)
    img_tensor_ = det.preprocess(img_)
    img_draw = draw_boxes(img_, boxes_)
    plt.imshow(img_draw)
    plt.show()
    a = 10
