from __future__ import annotations

import logging

from rich.logging import RichHandler
from torch import Tensor


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
