from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Sequence, Tuple, Type

from rich.logging import RichHandler
from torch import Tensor
from torch import device as Device
from torch import dtype as DType

from .interfaces import BatchInfo, RunnableTensor, TensorLike

logger = logging.getLogger(__name__)
logger.addHandler(RichHandler())
logger.setLevel(logging.DEBUG)

# So, it seems that torch's Tensor base class utilizes metaclass
# to pretend to be a parent of LongTensor, FloatTensor etc.
# Perhaps I'll be using the same paradigm.

# TODO: use the following class hook to generate a new subclass.
# https://pytorch.org/docs/stable/generated/torch.Tensor.as_subclass.html


class EagerTensor(Tensor):
    pass
