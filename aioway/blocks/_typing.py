# Copyright (c) RenChu Wang - All Rights Reserved

from numpy.typing import NDArray
from torch import Tensor

type Number = int | float
type Primitive = int | float | bool
type TensorNumber = Primitive | Tensor | NDArray
