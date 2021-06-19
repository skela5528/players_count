import enum
from typing import Optional, List

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.utils import draw_bounding_boxes


class ModelName(enum.IntEnum):
    ssd_lite = 1


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
    def _resize_boxes(boxes: torch.Tensor, origin_wh: tuple, net_dim: int) -> List[List[float]]:
        # resize center crop
        boxes = boxes.detach().cpu().numpy()
        width, height = origin_wh
        x_shift = (width - height) / 2
        scale_factor = height / net_dim

        # resize x
        boxes[:, 0] = boxes[:, 0] * scale_factor + x_shift
        boxes[:, 2] = boxes[:, 2] * scale_factor + x_shift
        # resize y
        boxes[:, 1] = boxes[:, 1] * scale_factor
        boxes[:, 3] = boxes[:, 3] * scale_factor
        # convert to list of Box
        return boxes

    def get_person_boxes(self, img: Image) -> dict:
        input_img = self.preprocess(img)
        input_img = input_img.unsqueeze(0)
        input_img = input_img.to(self.device)

        predictions = self.model(input_img)
        predictions = self._filter_predictions(predictions[0])
        print(predictions)
        return predictions


def draw_boxes(img: Image, boxes: list, color: tuple = (20, 20, 180)) -> np.ndarray:
    img = np.array(img)
    for box in boxes:
        left, top, right, bottom = [int(round(val)) for val in box]
        cv2.rectangle(img, (left, top), (right, bottom), color, 2)
    return img


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def vis_frame(img_tensor: torch.Tensor, boxes, scores):
    # min_score = .8
    # nms_iou_threshold = .9
    # indexes_nms = torchvision.ops.nms(predictions[0]['boxes'], predictions[0]['scores'], iou_threshold=nms_iou_threshold)

    score_strs = [str(round(s, 2)) for s in scores.detach().cpu().numpy()]
    img_vis = img_tensor * 255
    img_vis = img_vis.to(torch.uint8)
    img_draw = draw_bounding_boxes(img_vis, boxes, labels=score_strs)
    show(img_draw)


if __name__ == '__main__':
    p = '/home/cortica/Documents/my/git_personal/data/ml6/frames_png/00001.png'
    img_ = Image.open(p)

    det = Detector(ModelName.ssd_lite, .15, .5, (1920, 1080))
    predictions_ = det.get_person_boxes(img_)

    img_tensor_ = det.preprocess(img_)
    # vis_frame(img_tensor_, predictions_['boxes'], predictions_['scores'])

    # as
    boxes_ = det._resize_boxes(predictions_['boxes'], det.input_wh, det.net_dim)
    img_draw = draw_boxes(img_, boxes_)
    plt.imshow(img_draw)
    plt.show()
    a = 10
