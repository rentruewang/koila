# Copyright (c) RenChu Wang - All Rights Reserved

import enum
from enum import Enum


class DistType(Enum):
    CARTESIAN = enum.auto()
    INNER_PRODUCT = enum.auto()
