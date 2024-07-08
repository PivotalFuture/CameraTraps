# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import os
from glob import glob
import supervision as sv
from ultralytics.models import yolo

from typing import Any, Dict, Optional, List
import time
import torchvision
import torch
import numpy as np
from tqdm import tqdm


class BaseDetector:
    """
    Base detector class. This class provides utility methods for
    loading the model, generating results, and performing single and batch image detections.
    """

    # Placeholder class-level attributes to be defined in derived classes
    IMAGE_SIZE = None
    STRIDE = None
    CLASS_NAMES = None
    TRANSFORM = None

    def __init__(self, weights=None, device="cpu", url=None):
        """
        Initialize the base detector.

        Args:
            weights (str, optional):
                Path to the model weights. Defaults to None.
            device (str, optional):
                Device for model inference. Defaults to "cpu".
            url (str, optional):
                URL to fetch the model weights. Defaults to None.
        """
        self.model = None
        self.device = device
        self._load_model(weights, self.device, url)

    def _load_model(self, weights=None, device="cpu", url=None):
        """
        Load model weights.

        Args:
            weights (str, optional):
                Path to the model weights. Defaults to None.
            device (str, optional):
                Device for model inference. Defaults to "cpu".
            url (str, optional):
                URL to fetch the model weights. Defaults to None.
        Raises:
            Exception: If weights are not provided.
        """
        pass

    def results_generation(self, preds, img_id, id_strip=None):
        """
        Generate results for detection based on model predictions.

        Args:
            preds (numpy.ndarray):
                Model predictions.
            img_id (str):
                Image identifier.
            id_strip (str, optional):
                Strip specific characters from img_id. Defaults to None.

        Returns:
            dict: Dictionary containing image ID, detections, and labels.
        """
        pass

    def single_image_detection(
        self, img, img_size=None, img_path=None, conf_thres=0.2, id_strip=None
    ):
        """
        Perform detection on a single image.

        Args:
            img (torch.Tensor):
                Input image tensor.
            img_size (tuple):
                Original image size.
            img_path (str):
                Image path or identifier.
            conf_thres (float, optional):
                Confidence threshold for predictions. Defaults to 0.2.
            id_strip (str, optional):
                Characters to strip from img_id. Defaults to None.

        Returns:
            dict: Detection results.
        """
        pass

    def batch_image_detection(self, dataloader, conf_thres=0.2, id_strip=None):
        """
        Perform detection on a batch of images.

        Args:
            dataloader (DataLoader):
                DataLoader containing image batches.
            conf_thres (float, optional):
                Confidence threshold for predictions. Defaults to 0.2.
            id_strip (str, optional):
                Characters to strip from img_id. Defaults to None.

        Returns:
            list: List of detection results for all images.
        """
        pass


class YOLOV8Base(BaseDetector):
    """
    Base detector class for the new ultralytics YOLOV8 framework. This class provides utility methods for
    loading the model, generating results, and performing single and batch image detections.
    This base detector class is also compatible with all the new ultralytics models including YOLOV9,
    RTDetr, and more.
    """

    def __init__(self, weights=None, device="cpu", url=None):
        """
        Initialize the YOLOV8 detector.

        Args:
            weights (str, optional):
                Path to the model weights. Defaults to None.
            device (str, optional):
                Device for model inference. Defaults to "cpu".
            url (str, optional):
                URL to fetch the model weights. Defaults to None.
        """
        super(YOLOV8Base, self).__init__(weights=weights, device=device, url=url)

    def _load_model(self, weights=None, device="cpu", url=None):
        """
        Load the YOLOV8 model weights.

        Args:
            weights (str, optional):
                Path to the model weights. Defaults to None.
            device (str, optional):
                Device for model inference. Defaults to "cpu".
            url (str, optional):
                URL to fetch the model weights. Defaults to None.
        Raises:
            Exception: If weights are not provided.
        """

        self.predictor = yolo.detect.DetectionPredictor()
        # self.predictor.args.device = device # Will uncomment later
        self.predictor.args.imgsz = self.IMAGE_SIZE
        self.predictor.args.save = False  # Will see if we want to use ultralytics native inference saving functions.

        if weights:
            self.predictor.setup_model(weights)
        elif url:
            raise Exception("URL weights loading not ready for beta testing.")
        else:
            raise Exception("Need weights for inference.")

    def results_generation(self, preds, img_id, id_strip=None):
        """
        Generate results for detection based on model predictions.

        Args:
            preds (ultralytics.engine.results.Results):
                Model predictions.
            img_id (str):
                Image identifier.
            id_strip (str, optional):
                Strip specific characters from img_id. Defaults to None.

        Returns:
            dict: Dictionary containing image ID, detections, and labels.
        """
        xyxy = preds.boxes.xyxy.cpu().numpy()
        confidence = preds.boxes.conf.cpu().numpy()
        class_id = preds.boxes.cls.cpu().numpy().astype(int)

        results: Dict[str, Any] = {"img_id": str(img_id).strip(id_strip)}
        results["detections"] = sv.Detections(
            xyxy=xyxy, confidence=confidence, class_id=class_id
        )
        assert self.CLASS_NAMES is not None, "CLASS_NAMES not defined"
        results_ = []
        for detection in results["detections"]:
            assert len(detection) >= 5, f"Detection length is {len(detection)}"
            results_.append(
                (
                    detection[0][0],
                    detection[0][1],
                    detection[1],
                    detection[2],
                    detection[3],
                    detection[4],
                )
            )

        return results

    def single_image_detection(self, img=None, conf_thres=0.2, id_strip=None):
        """
        Perform detection on a single image.

        Args:
            img (torch.Tensor):
                an image tensor.
            conf_thres (float, optional):
                Confidence threshold for predictions. Defaults to 0.2.
            id_strip (str, optional):
                Characters to strip from img_id. Defaults to None.

        Returns:
            dict: Detection results.
        """

        self.predictor.args.batch = 1
        self.predictor.args.conf = conf_thres
        assert isinstance(img, torch.Tensor), "Input image must be a torch.Tensor"
        det_results = list(self.predictor.stream_inference(img.unsqueeze(0)))
        return self.results_generation(det_results[0], img, id_strip)

    def batch_image_detection(
        self, data_path, batch_size=16, conf_thres=0.2, id_strip=None, extension="JPG"
    ):
        """
        Perform detection on a batch of images.

        Args:
            data_path (str):
                Path containing all images for inference.
            batch_size (int, optional):
                Batch size for inference. Defaults to 16.
            conf_thres (float, optional):
                Confidence threshold for predictions. Defaults to 0.2.
            id_strip (str, optional):
                Characters to strip from img_id. Defaults to None.
            extension (str, optional):
                Image extension to search for. Defaults to "JPG"

        Returns:
            list: List of detection results for all images.
        """
        self.predictor.args.batch = batch_size
        self.predictor.args.conf = conf_thres
        img_list = glob(
            os.path.join(data_path, "**/*.{}".format(extension)), recursive=True
        )
        det_results = list(self.predictor.stream_inference(img_list))
        results = []
        for idx, preds in enumerate(det_results):
            results.append(self.results_generation(preds, img_list[idx], id_strip))
        return results


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (
            (img1_shape[1] - img0_shape[1] * gain) / 2,
            (img1_shape[0] - img0_shape[0] * gain) / 2,
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def box_area(box):
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
) -> List[torch.Tensor]:
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert (
        0 <= conf_thres <= 1
    ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.3 + 0.03 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output: List[torch.Tensor] = [torch.zeros((0, 6), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            assert isinstance(conf, torch.Tensor)
            assert isinstance(box, torch.Tensor)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                1, keepdim=True
            )  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


class YOLOV5Base:
    """
    Base detector class for YOLO V5. This class provides utility methods for
    loading the model, generating results, and performing single and batch image detections.
    """

    # Placeholder class-level attributes to be defined in derived classes
    IMAGE_SIZE = None
    STRIDE = None
    CLASS_NAMES: Optional[dict] = None
    TRANSFORM = None

    def __init__(self, weights=None, device="cpu"):
        """
        Initialize the YOLO V5 detector.

        Args:
            weights (str, optional):
                Path to the model weights. Defaults to None.
            device (str, optional):
                Device for model inference. Defaults to "cpu".
        """
        self.device = device
        model = self._load_model(weights, self.device)
        self.model = model.to(self.device)

    def _load_model(self, weights=None, device="cpu"):
        """
        Load the YOLO V5 model weights.

        Args:
            weights (str, optional):
                Path to the model weights. Defaults to None.
            device (str, optional):
                Device for model inference. Defaults to "cpu".
            url (str, optional):
                URL to fetch the model weights. Defaults to None.
        Raises:
            Exception: If weights are not provided.
        """
        if weights:
            import yolov5.models as models  # noqa

            checkpoint = torch.load(weights)
        else:
            raise Exception("Need weights for inference.")
        return (
            checkpoint["model"].float().fuse().eval().to(torch.device(device))
        )  # Convert to FP32 model

    def results_generation(self, preds, img_id, id_strip=None):
        """
        Generate results for detection based on model predictions.

        Args:
            preds (numpy.ndarray):
                Model predictions.
            img_id (str):
                Image identifier.
            id_strip (str, optional):
                Strip specific characters from img_id. Defaults to None.

        Returns:
            dict: Dictionary containing image ID, detections, and labels.
        """
        results: Dict[str, Any] = {"img_id": str(img_id).strip(id_strip)}
        results["detections"] = sv.Detections(
            xyxy=preds[:, :4], confidence=preds[:, 4], class_id=preds[:, 5].astype(int)
        )
        assert self.CLASS_NAMES is not None, "CLASS_NAMES not defined"
        results["labels"] = [
            f"{self.CLASS_NAMES[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _, _ in results["detections"]
        ]
        return results

    def single_image_detection(
        self, img, img_size=None, img_path=None, conf_thres=0.2, id_strip=None
    ):
        """
        Perform detection on a single image.

        Args:
            img (torch.Tensor):
                Input image tensor.
            img_size (tuple):
                Original image size.
            img_path (str):
                Image path or identifier.
            conf_thres (float, optional):
                Confidence threshold for predictions. Defaults to 0.2.
            id_strip (str, optional):
                Characters to strip from img_id. Defaults to None.

        Returns:
            dict: Detection results.
        """
        if img_size is None:
            img_size = img.permute(
                (1, 2, 0)
            ).shape  # We need hwc instead of chw for coord scaling
        preds = self.model(img.unsqueeze(0).to(self.device))[0]
        preds = torch.cat(
            non_max_suppression(prediction=preds, conf_thres=conf_thres), dim=0
        )
        preds[:, :4] = scale_coords(
            [self.IMAGE_SIZE] * 2, preds[:, :4], img_size
        ).round()

        return self.results_generation(preds.cpu().numpy(), img_path, id_strip)

    def batch_image_detection(self, dataloader, conf_thres=0.2, id_strip=None):
        """
        Perform detection on a batch of images.

        Args:
            dataloader (DataLoader):
                DataLoader containing image batches.
            conf_thres (float, optional):
                Confidence threshold for predictions. Defaults to 0.2.
            id_strip (str, optional):
                Characters to strip from img_id. Defaults to None.

        Returns:
            list: List of detection results for all images.
        """
        results = []
        total_preds = []
        total_paths = []
        total_img_sizes = []

        with tqdm(total=len(dataloader)) as pbar:
            for batch in dataloader:
                imgs, paths, sizes = batch
                imgs = imgs.to(self.device)
                total_preds.append(self.model(imgs)[0])
                total_paths.append(paths)
                total_img_sizes.append(sizes)
                pbar.update(1)

        total_preds = [
            non_max_suppression(prediction=pred.unsqueeze(0), conf_thres=conf_thres)[
                0
            ].numpy()
            for pred in torch.cat(total_preds, dim=0).cpu()
        ]
        total_paths = np.concatenate(total_paths, axis=0)
        total_img_sizes = np.concatenate(total_img_sizes, axis=0)

        # If there are size differences in the input images, use a for loop instead of matrix processing for scaling
        for pred, size, path in zip(total_preds, total_img_sizes, total_paths):
            pred[:, :4] = scale_coords([self.IMAGE_SIZE] * 2, pred[:, :4], size).round()
            results.append(self.results_generation(pred, path, id_strip))

        return results
