# Reference : https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/ReVision/__init__.py

# from typing import TYPE_CHECKING

# from transformers.utils import (
#     OptionalDependencyNotAvailable,
#     _LazyModule,
#     is_torch_available,
# )


# _import_structure = {"config": ["ReVisionConfig"]}


# try:
#     if not is_torch_available():
#         raise OptionalDependencyNotAvailable()
# except OptionalDependencyNotAvailable:
#     pass
# else:
#     _import_structure["revision_model"] = [
#         "ReVisionForConditionalGeneration",
#         "ReVisionPreTrainedModel",
#     ]
#     _import_structure["processor"] = ["ReVisionProcessor"]


# if TYPE_CHECKING:
#     from .config import ReVisionConfig

#     try:
#         if not is_torch_available():
#             raise OptionalDependencyNotAvailable()
#     except OptionalDependencyNotAvailable:
#         pass
#     else:
#         from .revision_model import (
#             ReVisionForConditionalGeneration,
#             ReVisionPreTrainedModel,
#         )
#         from .processor import ReVisionProcessor


# else:
#     import sys

#     sys.modules[__name__] = _LazyModule(
#         __name__, globals()["__file__"], _import_structure
#     )

from .config import ReVisionConfig
from .revision_model import ReVisionForConditionalGeneration, ReVisionPreTrainedModel
from .processor import ReVisionProcessor