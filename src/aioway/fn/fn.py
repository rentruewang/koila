# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import enum
import functools
import typing
from collections import abc as cabc

from aioway import ctx

__all__ = ["Fn", "FnState"]


class FnState(enum.Enum):
    "The status of a `Later` object."

    PENDING = enum.auto()
    "The object is pending evaluation."

    DONE = enum.auto()
    "The object is evaluated."


class Fn[T](abc.ABC):
    """
    `Fn`s represent computation that shall be done later.

    Like Haskell's thunks, once evaluated,
    the value is stored in the `Fn` itself and never re-evaluated.
    The value shall be gone during GC.

    I was going to go for `Op` but it's used a lot in `torch`.
    """

    __match_args__: typing.ClassVar[tuple[str, ...]]

    def __repr__(self):
        name = self._name()
        return f"{name}<{self.state}>"

    @typing.final
    def do(self) -> T:
        """
        Perform the computation that is represented by this `Fn`.

        This is the public function that parent `Fn`s should call,
        when the want to request the values of an `Fn`.

        It handles caching in the normal case, so repeated calling the function means that
        the expensive computation (defined within `forward`) would only be called once.

        When fake mode is enabled, it calls `preview` for a fake tensor,
        which is a preview for the normal computation to save computation cost.

        The reason this is modal with `fake.is_enabled()` as a toggle,
        to make sure `preview` and `forward` can use the same codepath as much as possible,
        in the default case `preview` is `forward` with fake mode on.
        """

        if ctx.enabled_fake_mode():
            return self.preview()

        else:
            return self.__forward_cache()

    @ctx.fake_mode_func
    def preview(self) -> T:
        """
        The `preview` function generates a "preview" for the `Tensor` that would be generated.
        Should recursively call the dependent `Fn.do` functions.

        The result type (`FakeTensor`) is used as a worst case analysis of the original `Tensor`.

        In most cases (non leaf operators), this method is just a clone of `forward`,
        which is the default implementation of this function.

        In the following cases it must be modified:

        1. Source tensors, `forward` won't be `FakeTensor`, so conversion is needed.
        2. Operators that cannot be supported by `torch` e.g. boolean  masking.
        """

        result = self.forward()
        assert ctx.is_fake_tensor(result)
        return result

    @abc.abstractmethod
    def forward(self) -> T:
        raise NotImplementedError

    @abc.abstractmethod
    def deps(self) -> tuple[Fn[typing.Any], ...]:
        """
        The `Fn`s that must be evaluated before we can evaluate the current `Fn`.

        Calling `do` on the current `Fn` would recursively
        """

        raise NotImplementedError

    @property
    def is_leaf(self) -> bool:
        "Whether or not the thunk is dependent on other thunks. If not, it's a leaf."

        return not self.deps()

    @functools.cached_property
    def __forward_cache(self):
        return FnCache(self.forward)

    @property
    def done(self) -> bool:
        "Whether or not this is done."

        return self.__forward_cache.is_hit

    @property
    def state(self) -> FnState:
        "The state of the `Fn`. Would be an instance of `FnState` enum."

        return FnState.DONE if self.done else FnState.PENDING

    def _name(self) -> str:
        """
        The name of the `Fn` used in `repr`.
        """

        return type(self).__name__


_PENDING = object()
"The object signifying a status of pending. This is a `object()` s.t. `FnCache` can store `None`."


@typing.final
class FnCache[T]:
    """
    The cacher for `TensorFn.forward`.

    The reason we use this boilerplate over directly using `functools.cache`,
    `functools.cached_property`, or having a saved `.__result` member for instance,
    is because this is the least assuming.

    `functools.cache` assumes that `self` is hashable.
    `functools.cached_property` cannot inspect whether we have evaluated it or not.
    `.__result` member assumes subclass calls `__init__` properly.

    Since this is saved in a `functools.cached_property`, it can be used on unhashable types,
    yet support inspecting whether we called it or not, and does not need to call `__init__`.
    """

    def __init__(self, func: cabc.Callable[[], T]) -> None:
        self._result: object = _PENDING
        self._func = func

    def __call__(self) -> T:
        if self._result is _PENDING:
            self._result = self._func()

        return typing.cast(T, self._result)

    @property
    def is_hit(self) -> bool:
        "Returns if the cache is previously called."

        return self._result is not _PENDING
