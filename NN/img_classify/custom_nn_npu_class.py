from transformers import AutoModelForImageClassification
import intel_npu_acceleration_library as npu_lib
from functools import partialmethod
from typing import Type, Any, Tuple, Optional
import hashlib
import torch
import os

class NPUModelForImageClassification:
    """NPU wrapper for AutoModelForImageClassification.

    Attrs:
        from_pretrained: Load a pretrained model
    """

    from_pretrained = partialmethod(
        npu_lib.NPUModel.from_pretrained, transformers_class=AutoModelForImageClassification
    )
