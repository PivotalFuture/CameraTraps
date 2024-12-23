# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""YoloV5 base detector class."""

from typing import Optional, List, Union, Tuple, Any, Callable
import numpy as np
from tqdm import tqdm
from PIL import Image
from supervision.detection.core import Detections
import torch
from torch.utils.data import DataLoader
from torch.hub import load_state_dict_from_url
from yolov5.utils.general import non_max_suppression, scale_coords
from ..base_detector import BaseDetector, DetectionResult
from ....data import transforms as pw_trans
from ....data import datasets as pw_data


class MegaDetectorTransform(pw_trans.MegaDetector_v5_Transform):
    def __init__(self, target_size: Union[Tuple[int, int], int], stride: int) -> None:
        super().__init__(
            target_size=int(
                target_size[0] if isinstance(target_size, tuple) else target_size
            ),
            stride=stride,
        )


class YOLOV5Base(BaseDetector):
    """Base detector class for YOLO V5."""

    model: Any

    def __init__(
        self,
        weights: Optional[str] = None,
        device: str = "cpu",
        url: Optional[str] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        """Initialize YOLO V5 detector."""
        self.transform = transform
        super().__init__(weights=weights, device=device, url=url)
        self.load_model(weights, device, url)

    def load_model(
        self,
        weights: Optional[str] = None,
        device: str = "cpu",
        url: Optional[str] = None,
    ) -> None:
        if weights:
            checkpoint = torch.load(weights, map_location=device)
        elif url:
            checkpoint = load_state_dict_from_url(url, map_location=device)
        else:
            raise Exception("Need weights for inference.")

        self.model = checkpoint["model"].float().fuse().eval().to(device)

        if (
            not self.transform
            and self.IMAGE_SIZE is not None
            and self.STRIDE is not None
        ):
            self.transform = MegaDetectorTransform(
                target_size=self.IMAGE_SIZE, stride=self.STRIDE
            )

    def results_generation(
        self, preds: np.ndarray, img_id: str, id_strip: Optional[str] = None
    ) -> DetectionResult:
        results: DetectionResult = {
            "img_id": str(img_id).strip(id_strip) if id_strip else str(img_id)
        }

        if len(preds) > 0 and self.CLASS_NAMES is not None:
            if preds.size > 0:  # Check if array is not empty
                detections = Detections(
                    xyxy=preds[:, :4].astype(np.float32),
                    confidence=preds[:, 4].astype(np.float32),
                    class_id=preds[:, 5].astype(int),
                )
                if detections.confidence is None or detections.class_id is None:
                    raise ValueError("Invalid detections")

                results["detections"] = detections

                results["labels"] = [
                    f"{self.CLASS_NAMES[class_id]} {confidence:0.2f}"
                    for confidence, class_id in zip(
                        detections.confidence, detections.class_id
                    )
                ]

        return results

    def single_image_detection(
        self,
        img: Union[str, np.ndarray],
        det_conf_thres: float = 0.2,
        img_path: Optional[str] = None,
        img_size: Optional[Tuple[int, int]] = None,
        id_strip: Optional[str] = None,
    ) -> DetectionResult:
        if isinstance(img, str):
            if img_path is None:
                img_path = img
            img = np.array(Image.open(img).convert("RGB"))

        if self.transform is None:
            raise ValueError("Transform not initialized")

        transformed_img = self.transform(img)
        curr_img_size = img_size if img_size else img.shape

        with torch.no_grad():
            preds = self.model(transformed_img.unsqueeze(0).to(self.device))[0]
            preds = non_max_suppression(preds, conf_thres=det_conf_thres)

        if preds[0].size(0) > 0:
            preds_np = preds[0].cpu().numpy()
            preds_np[:, :4] = scale_coords(
                [self.IMAGE_SIZE] * 2
                if isinstance(self.IMAGE_SIZE, int)
                else self.IMAGE_SIZE,
                preds_np[:, :4],
                curr_img_size,
            ).round()
            return self.results_generation(preds_np, img_path or "", id_strip)

        return {"img_id": str(img_path or "")}

    def batch_image_detection(
        self,
        data_path: str,
        batch_size: int = 16,
        det_conf_thres: float = 0.2,
        id_strip: Optional[str] = None,
    ) -> List[DetectionResult]:
        dataset = pw_data.DetectionImageFolder(
            data_path,
            transform=self.transform,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
            drop_last=False,
        )

        results: List[DetectionResult] = []
        with tqdm(total=len(loader)) as pbar:
            for _, (imgs, paths, sizes) in enumerate(loader):
                imgs = imgs.to(self.device)
                with torch.no_grad():
                    predictions = self.model(imgs)[0]
                    predictions = non_max_suppression(
                        predictions, conf_thres=det_conf_thres
                    )

                for i, pred in enumerate(predictions):
                    if pred.size(0) == 0:
                        results.append({"img_id": str(paths[i])})
                        continue

                    pred_np = pred.cpu().numpy()
                    size = sizes[i].numpy()
                    path = paths[i]

                    pred_np[:, :4] = scale_coords(
                        [self.IMAGE_SIZE] * 2
                        if isinstance(self.IMAGE_SIZE, int)
                        else self.IMAGE_SIZE,
                        pred_np[:, :4],
                        size,
                    ).round()

                    result = self.results_generation(pred_np, path, id_strip)
                    result["normalized_coords"] = [
                        [x1 / size[1], y1 / size[0], x2 / size[1], y2 / size[0]]
                        for x1, y1, x2, y2 in pred_np[:, :4]
                    ]
                    results.append(result)
                pbar.update(1)

        return results
