# Copyright (c) Microsoft Corporation. All rights reserved.
"""YOLOv8 base detector class."""

import os
from typing import Optional, List, Union, Callable, cast, Tuple, Any
import numpy as np
from PIL import Image
import wget
import torch
from ultralytics.models import yolo
from ultralytics.engine.results import Results
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..base_detector import BaseDetector, DetectionResult
from ....data import transforms as pw_trans
from ....data import datasets as pw_data


class MegaDetectorTransform(pw_trans.MegaDetector_v5_Transform):
    """Transform class for MegaDetector input preprocessing."""

    def __init__(self, target_size: Union[Tuple[int, int], int], stride: int) -> None:
        target_int = target_size[0] if isinstance(target_size, tuple) else target_size
        super().__init__(target_size=target_int, stride=stride)


class YOLOV8Base(BaseDetector):
    """Base detector class for YOLO V8."""

    predictor: yolo.detect.DetectionPredictor

    def __init__(
        self,
        weights: Optional[str] = None,
        device: str = "cpu",
        url: Optional[str] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        self.transform = transform
        super().__init__(weights=weights, device=device, url=url)
        self.load_model(weights, device, url)

    def load_model(
        self,
        weights: Optional[str] = None,
        device: str = "cpu",
        url: Optional[str] = None,
    ) -> None:
        try:
            self.predictor = yolo.detect.DetectionPredictor()
            self.predictor.args.imgsz = self.IMAGE_SIZE
            self.predictor.args.save = False
            self.predictor.args.half = False  # Ensure full precision

            if weights:
                self.predictor.setup_model(weights)
            elif url:
                checkpoint_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
                checkpoint_path = os.path.join(checkpoint_dir, "MDV6b-yolov9c.pt")

                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    weights = wget.download(url, out=checkpoint_dir)
                else:
                    weights = checkpoint_path

                self.predictor.setup_model(weights)
            else:
                raise Exception("Need weights or URL for model initialization")

            # Override predictor's preprocess method to ensure float32
            original_preprocess = self.predictor.preprocess

            def preprocess_wrapper(*args: Any, **kwargs: Any) -> torch.Tensor:
                batch = original_preprocess(*args, **kwargs)
                if isinstance(batch, torch.Tensor):
                    batch = batch.float()
                return batch

            self.predictor.preprocess = preprocess_wrapper

            if (
                not self.transform
                and self.IMAGE_SIZE is not None
                and self.STRIDE is not None
            ):
                self.transform = MegaDetectorTransform(
                    target_size=self.IMAGE_SIZE, stride=self.STRIDE
                )

            # Ensure model is float32 before moving to device
            if hasattr(self.predictor, "model"):
                assert isinstance(self.predictor.model, torch.nn.Module)
                self.predictor.model = self.predictor.model.float().to(device)

            self.predictor.device = device

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}") from e

    def _process_boxes(
        self, preds: Results
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process detection boxes from model predictions."""
        assert preds.boxes is not None, "No detections found in predictions"

        xyxy = preds.boxes.xyxy
        confidence = preds.boxes.conf
        class_id = preds.boxes.cls

        # Convert tensors to numpy arrays if needed
        if isinstance(xyxy, torch.Tensor):
            xyxy = xyxy.cpu().numpy()
        if isinstance(confidence, torch.Tensor):
            confidence = confidence.cpu().numpy()
        if isinstance(class_id, torch.Tensor):
            class_id = class_id.cpu().numpy()

        return xyxy, confidence, class_id.astype(int)

    def results_generation(
        self,
        preds: Results,
    ) -> List[DetectionResult]:
        """Generate detection results from model predictions."""
        if preds.boxes is None:
            return []

        try:
            xyxy, confidence, class_id = self._process_boxes(preds)
            results: List[DetectionResult] = []

            if self.CLASS_NAMES is not None:
                for conf, cls, box in zip(confidence, class_id, xyxy):
                    if cls < len(self.CLASS_NAMES):
                        results.append(
                            DetectionResult(
                                bbox=box.tolist(),
                                confidence=float(conf),
                                label=f"{self.CLASS_NAMES[cls]} {conf:0.2f}",
                            )
                        )

            return results
        except Exception as e:
            print(f"Warning: Error generating results: {str(e)}")
            return []

    def single_image_detection(
        self,
        img: Union[str, np.ndarray],
        det_conf_thres: float = 0.2,
        img_path: Optional[str] = None,
    ) -> List[DetectionResult]:
        """Perform detection on a single image."""
        try:
            if isinstance(img, str):
                if img_path is None:
                    img_path = img
                img = np.array(Image.open(img).convert("RGB"))

            # Ensure image is float32
            if isinstance(img, np.ndarray):
                img = img.astype(np.float32)

            self.predictor.args.batch = 1
            self.predictor.args.conf = det_conf_thres

            det_results = list(self.predictor.stream_inference([img]))
            result = cast(Results, det_results[0])

            return self.results_generation(result)

        except Exception as e:
            print(f"Warning: Error in single image detection: {str(e)}")
            return []

    def batch_image_detection(
        self,
        batch: Tuple[np.ndarray, ...] | str,
        batch_size: int = 16,
        det_conf_thres: float = 0.2,
    ) -> List[List[DetectionResult]]:
        """Perform detection on a batch of images."""
        self.predictor.args.batch = batch_size
        self.predictor.args.conf = det_conf_thres

        try:
            if isinstance(batch, str):
                dataset = pw_data.DetectionImageFolder(batch, transform=self.transform)

            elif isinstance(batch, tuple):
                dataset = pw_data.DetectionImageTuple(batch, transform=self.transform)

            else:
                raise ValueError("Invalid batch input")

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=0,
                drop_last=False,
            )

            results: List[List[DetectionResult]] = []
            with tqdm(total=len(loader)) as pbar:
                for _, (_, paths, _) in enumerate(loader):
                    try:
                        det_results = self.predictor.stream_inference(paths)

                        for preds in det_results:
                            preds = cast(Results, preds)
                            if preds.orig_shape is None:
                                results.append([])
                                continue

                            size = preds.orig_shape
                            result = self.results_generation(preds)

                            # Normalize bounding box coordinates
                            for idx, res in enumerate(result):
                                bbox = res.bbox
                                bbox = (
                                    bbox[0] / size[1],
                                    bbox[1] / size[0],
                                    bbox[2] / size[1],
                                    bbox[3] / size[0],
                                )
                                res.bbox = bbox
                                result[idx] = res

                            results.append(result)

                    except Exception as e:
                        print(f"Warning: Error processing batch: {str(e)}")
                        results.extend([[] for _ in range(len(paths))])

                    pbar.update(1)

            return results

        except Exception as e:
            print(f"Warning: Error in batch detection: {str(e)}")
            return []
