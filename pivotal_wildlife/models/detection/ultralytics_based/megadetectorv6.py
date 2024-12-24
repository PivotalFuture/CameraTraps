from .yolov8_base import YOLOV8Base
from typing import Literal

__all__ = ["MegaDetectorV6"]


class MegaDetectorV6(YOLOV8Base):
    """
    MegaDetectorV6 is a specialized class derived from the YOLOV8Base class
    that is specifically designed for detecting animals, persons, and vehicles.

    Attributes:
        CLASS_NAMES (dict): Mapping of class IDs to their respective names.
        IMAGE_SIZE (int): The standard image size used during training.
        STRIDE (int): Stride value used in the detector.
    """

    CLASS_NAMES = ["animal", "person", "vehicle"]
    STRIDE = 32  # Added STRIDE attribute

    def __init__(
        self,
        weights: str | None = None,
        device: str = "cpu",
        pretrained: bool | None = True,
        version: Literal["yolov9c", "rtdetrl"] | None = "yolov9c",
    ):
        """
        Initializes the MegaDetectorV6 model with the option to load pretrained weights.

        Args:
            weights (str, optional): Path to the weights file.
            device (str, optional): Device to load the model on (e.g., "cpu" or "cuda"). Default is "cpu".
            pretrained (bool, optional): Whether to load the pretrained model. Default is True.
            version (str, optional): Version of the model to load. Default is 'yolov9c'.
        """

        if pretrained:
            if version == "yolov9c":
                self.IMAGE_SIZE = 640
                url = "https://zenodo.org/records/13357337/files/MDV6b-yolov9c.pt?download=1"
            elif version == "rtdetrl":
                self.IMAGE_SIZE = 640
                url = None
            else:
                raise ValueError("Select a valid model version: yolov9c or rtdetrl")
        else:
            url = None

        super(MegaDetectorV6, self).__init__(weights=weights, device=device, url=url)
