from __future__ import annotations

import functools
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    runtime_checkable,
)

T = TypeVar("T", covariant=True)
V = TypeVar("V", contravariant=True)


@runtime_checkable
class Runnable(Protocol[T]):
    @abstractmethod
    def run(self) -> T:
        ...


@dataclass(frozen=True)
class Lazy(Runnable[T]):
    data: T | Runnable[T]

    def __init__(self, data: T | Runnable[T] | Lazy[T]) -> None:
        if isinstance(data, Lazy):
            object.__setattr__(self, "data", data.data)
        else:
            object.__setattr__(self, "data", data)

    def run(self) -> T:
        data = self.data
        if isinstance(data, Runnable):
            return data.run()
        return data


@dataclass(frozen=True)
class LazyFunction(Generic[T, V]):
    func: Callable[..., T]

    def __call__(self, *args: Any, **kwargs: Any) -> Lazy[T]:
        return Lazy(Evaluation(self.func, *args, **kwargs))

    def __get__(self, obj: V, objtype: Type[V]) -> Callable[..., Lazy[T]]:
        assert isinstance(obj, objtype), [type(obj), objtype]
        if obj is None:
            return self
        else:
            return functools.partial(self, obj)


@dataclass
class Evaluation(Runnable[T]):
    func: Callable[..., T]
    args: Tuple[Lazy[Any], ...]
    kwargs: Dict[str, Lazy[Any]]

    def __init__(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> None:
        self.func = func
        self.args = tuple(Lazy(arg) for arg in args)
        self.kwargs = dict((k, Lazy(v)) for (k, v) in kwargs.items())

    def run(self) -> T:
        real_args = tuple(arg.run() for arg in self.args)
        real_kwargs = dict((k, v.run()) for (k, v) in self.kwargs.items())
        return self.func(*real_args, **real_kwargs)
