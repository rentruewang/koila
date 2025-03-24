# Copyright (c) RenChu Wang - All Rights Reserved

"""
``Exec``s
#########

``Exec`` are similar to ``torch``'s ``IterableDataset``, except batched.

This design decision is made because we would like to enable lazy / iterator processing,
and if we directly follow the abstraction of ``IterableDataset``,
we have to process the tensor representation of the items 1 by 1, which can be inefficient.
"""

from ._data_loader import *
from .binary import *
from .execs import *
from .inputs import *
from .unary import *
