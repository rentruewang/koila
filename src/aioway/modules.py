# Copyright (c) AIoWay Authors - All Rights Reserved

"The preview module protocol definition."

import functools
from collections.abc import Callable

from torch import Tensor
from torch.nn import Module as NnModule

from aioway import fake
from aioway._previews import Attr
from aioway._signs import Signature
from aioway._tracking import ModuleApiTracker, logging

__all__ = ["Module"]

LOGGER = logging.get_logger(__name__)


class Module[**P, T: NnModule]:
    """
    Preview informs us how an `nn.Module` would be behave without initializing it.

    Example:
        `Module(torch.nn.Linear, in_features=1, out_features=2)`.
        Note that the arguments following the first type argument will be passed into the `nn.Module` type,
        so they are exactly as those on the `torch` documentation.
    """

    def __init__(
        self, nn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
    ) -> None:
        self._module = nn
        self._args = args
        self._kwargs = kwargs

    @functools.cached_property
    @fake.enable_func
    def fake_module(self) -> T:
        return self._module(*self._args, **self._kwargs)

    @functools.cached_property
    def real_module(self) -> T:
        return self._module(*self._args, **self._kwargs)

    def preview(self, attr: Attr, /) -> Attr:
        """
        Transforms the input attribute into an attribute describing the output.

        Returns `NotImplemented` when the input `Attr` is incompatible (usually device and dtype).
        """

        with self._tracker()("preview", Signature(Attr, Attr)):
            return self._preview(attr)

    @fake.enable_func
    def _preview(self, attr: Attr) -> Attr:
        tensor = attr.to_tensor()
        result: Tensor = self.fake_module(tensor)
        assert fake.is_fake_tensor(result), "Function is running under fake mode."
        return Attr.from_tensor(result)

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Do a forward pass on the input `tensor`.
        """

        with self._tracker()("forward", Signature(Tensor, Tensor)):
            return self.real_module(tensor)

    @classmethod
    def _tracker(cls) -> ModuleApiTracker:
        return ModuleApiTracker(lambda: cls)
