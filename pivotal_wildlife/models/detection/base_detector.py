# Copyright (c) Microsoft Corporation. All rights reserved.
"""Base detector class."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Union, Tuple, Any, TypeVar
import numpy as np
from torch import nn
from ultralytics.engine.results import Results
from supervision.detection.core import Detections

R = TypeVar("R", Results, np.ndarray)
DetectionResult = Dict[
    str, Union[str, Detections, List[str], List[List[float]], np.ndarray]
]


class BaseDetector(nn.Module, ABC):
    """Base detector class."""

    IMAGE_SIZE: Optional[Union[Tuple[int, int], int]] = None
    STRIDE: Optional[int] = None
    CLASS_NAMES: Optional[List[str]] = None
    TRANSFORM: Optional[Any] = None

    def __init__(
        self,
        weights: Optional[str] = None,
        device: str = "cpu",
        url: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.device = device
        if weights is not None or url is not None:
            self.load_model(weights, device, url)

    @abstractmethod
    def load_model(
        self,
        weights: Optional[str] = None,
        device: str = "cpu",
        url: Optional[str] = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def results_generation(
        self, preds: R, img_id: str, id_strip: Optional[str] = None
    ) -> DetectionResult:
        raise NotImplementedError

    @abstractmethod
    def single_image_detection(
        self,
        img: Union[str, np.ndarray],
        det_conf_thres: float = 0.2,
        img_path: Optional[str] = None,
        img_size: Optional[Tuple[int, int]] = None,
        id_strip: Optional[str] = None,
    ) -> DetectionResult:
        raise NotImplementedError

    @abstractmethod
    def batch_image_detection(
        self,
        data_path: str,
        batch_size: int = 16,
        det_conf_thres: float = 0.2,
        id_strip: Optional[str] = None,
    ) -> List[DetectionResult]:
        raise NotImplementedError
