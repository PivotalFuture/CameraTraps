# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from .base_detector import YOLOV5Base

__all__ = [
    "MegaDetectorV5",
]


class MegaDetectorV5(YOLOV5Base):
    """
    MegaDetectorV5 is a specialized class derived from the YOLOV5Base class
    that is specifically designed for detecting animals, persons, and vehicles.

    Attributes:
        IMAGE_SIZE (int): The standard image size used during training.
        STRIDE (int): Stride value used in the detector.
        CLASS_NAMES (dict): Mapping of class IDs to their respective names.
    """

    IMAGE_SIZE = 1280  # image size used in training
    STRIDE = 64
    CLASS_NAMES = {0: "animal", 1: "person", 2: "vehicle"}

    def __init__(self, weights=None, device="cpu"):
        """
        Initializes the MegaDetectorV5 model with the option to load pretrained weights.

        Args:
            weights (str, optional): Path to the weights file.
            device (str, optional): Device to load the model on (e.g., "cpu" or "cuda"). Default is "cpu".
        """

        super(MegaDetectorV5, self).__init__(weights=weights, device=device)
