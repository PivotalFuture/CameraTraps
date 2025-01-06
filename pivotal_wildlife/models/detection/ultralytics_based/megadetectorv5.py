# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from .yolov5_base import YOLOV5Base
from typing import Literal

__all__ = ["MegaDetectorV5"]


class MegaDetectorV5(YOLOV5Base):
    """
    MegaDetectorV5 is a specialized class derived from the YOLOV5Base class
    that is specifically designed for detecting animals, persons, and vehicles.

    Attributes:
        IMAGE_SIZE (int): The standard image size used during training.
        STRIDE (int): Stride value used in the detector.
        CLASS_NAMES (dict): Mapping of class IDs to their respective names.
    """

    IMAGE_SIZE = 1280
    STRIDE = 64
    CLASS_NAMES = ["animal", "person", "vehicle"]

    def __init__(
        self,
        weights: str | None = None,
        device: str = "cpu",
        pretrained: bool | None = True,
        version: Literal["a", "b"] | None = "a",
    ):
        """
        Initializes the MegaDetectorV5 model with the option to load pretrained weights.

        Args:
            weights (str, optional): Path to the weights file.
            device (str, optional): Device to load the model on (e.g., "cpu" or "cuda"). Default is "cpu".
            pretrained (bool, optional): Whether to load the pretrained model. Default is True.
            version (str, optional): Version of the MegaDetectorV5 model to load. Default is "a".
        """

        if pretrained:
            if version == "a":
                url = (
                    "https://zenodo.org/records/13357337/files/md_v5a.0.0.pt?download=1"
                )
            elif version == "b":
                url = "https://zenodo.org/records/10023414/files/MegaDetector_v5b.0.0.pt?download=1"
            else:
                url = None
        else:
            url = None

        super(MegaDetectorV5, self).__init__(weights=weights, device=device, url=url)