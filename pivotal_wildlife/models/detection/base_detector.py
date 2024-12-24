# Copyright (c) Microsoft Corporation. All rights reserved.
"""Base detector class."""

from abc import ABC, abstractmethod
from typing import Optional, List, Union, Tuple, Any, TypeVar
from dataclasses import dataclass
import numpy as np
from torch import nn
from ultralytics.engine.results import Results

R = TypeVar("R", Results, np.ndarray)


@dataclass
class DetectionResult:
    bbox: Tuple[float, float, float, float]
    confidence: float
    label: str


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
        self,
        preds: R,
    ) -> List[DetectionResult]:
        raise NotImplementedError

    @abstractmethod
    def single_image_detection(
        self,
        img: Union[str, np.ndarray],
        det_conf_thres: float = 0.2,
        img_path: Optional[str] = None,
    ) -> List[DetectionResult]:
        raise NotImplementedError

    @abstractmethod
    def batch_image_detection(
        self,
        data_path: str,
        batch_size: int = 16,
        det_conf_thres: float = 0.2,
    ) -> List[List[DetectionResult]]:
        raise NotImplementedError
