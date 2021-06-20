import enum
from typing import Optional, List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, retinanet_resnet50_fpn


class ModelName(enum.IntEnum):
    ssd_lite = 1
    retina_net = 2


class Detector:
    def __init__(self, model_name: ModelName, score_thresh: float, nms_thresh: float, input_wh: tuple, device: str = 'cuda'):
        self.input_wh = input_wh
        self.device = device
        self.nms_thresh = nms_thresh
        self.net_dim: int = 320

        self.model = self._load_model(model_name, score_thresh, nms_thresh, device)
        self.preprocess = self._get_img_transforms(self.net_dim)

    @staticmethod
    def _get_img_transforms(net_in_size: int, crop_size: int = 960) -> transforms.Compose:
        img_transforms = transforms.Compose([
            transforms.FiveCrop(crop_size),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Resize(size=net_in_size, interpolation=transforms.InterpolationMode.BICUBIC),
        ])
        return img_transforms

    @staticmethod
    def _load_model(model_name: ModelName, score_thresh: float, nms_thresh: float, device: str) -> torch.nn.Module:
        if model_name == ModelName.ssd_lite:
            model = ssdlite320_mobilenet_v3_large(pretrained=True, pretrained_backbone=True, score_thresh=score_thresh, nms_thresh=nms_thresh)
        elif model_name == ModelName.retina_net:
            model = retinanet_resnet50_fpn(pretrained=True, pretrained_backbone=True, score_thresh=score_thresh, nms_thresh=nms_thresh)
        else:
            raise ValueError
        model.to(device)
        model.eval()
        return model

    @staticmethod
    def _filter_predictions(predictions: dict, white_list_labels: tuple = (1,)) -> Optional[dict]:
        white_list_labels = list(white_list_labels)
        indexes_keep = [i for i, label in enumerate(predictions['labels']) if label in white_list_labels]
        indexes_keep = torch.tensor(indexes_keep)
        result = None
        if len(indexes_keep):
            result = {key: values[indexes_keep] for key, values in predictions.items()}
        return result

    @staticmethod
    def _resize_boxes_five_crops(predictions: List[dict], origin_wh: tuple, net_dim: int, crop_size: int = 960) -> (List[list], List[float]):
        # tl, tr, bl, br, center - crops order

        # calculate shifts
        width, height = origin_wh
        scale_factor = crop_size / net_dim
        dx = width - crop_size
        dy = height - crop_size
        x_shifts = (0, dx, 0, dx, dx / 2)
        y_shifts = (0, 0, dy, dy, dy / 2)

        # resize each crop
        boxes, scores = [], []
        for ind, crop_pred in enumerate(predictions):
            crop_boxes = to_numpy(crop_pred['boxes'])
            crop_scores = to_numpy(crop_pred['scores'])

            # resize x inplace
            crop_boxes[:, 0] = crop_boxes[:, 0] * scale_factor + x_shifts[ind]
            crop_boxes[:, 2] = crop_boxes[:, 2] * scale_factor + x_shifts[ind]

            # resize y inplace
            crop_boxes[:, 1] = crop_boxes[:, 1] * scale_factor + y_shifts[ind]
            crop_boxes[:, 3] = crop_boxes[:, 3] * scale_factor + y_shifts[ind]

            # to list of lists
            crop_boxes = crop_boxes.tolist()
            boxes.extend(crop_boxes)
            scores.extend(crop_scores)
        return boxes, scores

    def get_person_boxes(self, img: Image) -> (List[list], List[float]):
        # prepare input for net
        input_img = self.preprocess(img)
        input_img = input_img.to(self.device)

        # predict
        predictions = self.model(input_img)

        # parse prediction
        predictions = [self._filter_predictions(p) for p in predictions]
        boxes, scores = self._resize_boxes_five_crops(predictions, self.input_wh, self.net_dim)

        # nms
        keep_ids = nms(boxes, scores, self.nms_thresh, min_mode=True)
        boxes, scores = [boxes[i] for i in keep_ids], [scores[i] for i in keep_ids]
        return boxes, scores


def to_numpy(values: torch.Tensor) -> np.ndarray:
    return values.detach().cpu().numpy()


def nms(boxes: List[list], scores: List[float], overlap_threshold: float, min_mode: bool = False) -> List[int]:
    """Not Maximum Suppression."""

    scores = np.array(scores)
    x1 = np.array([b[0] for b in boxes])
    y1 = np.array([b[1] for b in boxes])
    x2 = np.array([b[2] for b in boxes])
    y2 = np.array([b[3] for b in boxes])

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep_indices = []
    while order.size > 0:
        keep_indices.append(order[0])
        xx1 = np.maximum(x1[order[0]], x1[order[1:]])
        yy1 = np.maximum(y1[order[0]], y1[order[1:]])
        xx2 = np.minimum(x2[order[0]], x2[order[1:]])
        yy2 = np.minimum(y2[order[0]], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        if min_mode:
            ovr = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            ovr = inter / (areas[order[0]] + areas[order[1:]] - inter)
        inds = np.where(ovr <= overlap_threshold)[0]
        order = order[inds + 1]
    return keep_indices


def debug_detector():
    from matplotlib import pyplot as plt
    from misc import draw_boxes

    sample_img_path = '../data/00001.jpg'
    img = Image.open(sample_img_path)

    det = Detector(ModelName.ssd_lite, .15, .5, (1920, 1080))
    boxes, scores = det.get_person_boxes(img)
    img_draw = draw_boxes(img, boxes)
    plt.imshow(img_draw)
    plt.show()


if __name__ == '__main__':
    debug_detector()
