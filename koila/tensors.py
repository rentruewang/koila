from __future__ import annotations

import builtins
import dataclasses as dcls
import functools
import logging
from dataclasses import dataclass
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    NamedTuple,
    NoReturn,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    final,
    overload,
)

import torch
from rich.logging import RichHandler
from torch import Tensor, cuda
from torch import device as Device
from torch import dtype as DType

from . import gpus, interfaces, prepasses
from .errors import UnsupportedError
from .interfaces import BatchInfo, Runnable, RunnableTensor, TensorLike
from .prepasses import PrePass, PrePassFunc

T = TypeVar("T")
V = TypeVar("V", contravariant=True)

logger = logging.getLogger(__name__)
logger.addHandler(RichHandler())


@dataclass(frozen=True)
class LazyFunction(Generic[V]):
    func: Callable[..., Tensor]
    prepass_func: PrePassFunc

    def __call__(self, *args: Any, **kwargs: Any) -> LazyTensor:
        lazy_args = tuple(lazy(arg) for arg in args)
        lazy_kwargs = dict((k, lazy(v)) for (k, v) in kwargs.items())
        prepass = self.prepass_func(*args, **kwargs)
        return LazyTensor(Evaluation(self.func, prepass, *lazy_args, **lazy_kwargs))

    def __get__(self, obj: V, objtype: Type[V]) -> Callable[..., LazyTensor]:
        assert isinstance(obj, objtype), [type(obj), objtype]
        if obj is None:
            return self
        else:
            return functools.partial(self, obj)


@final
@dataclass(init=False)
class Evaluation(RunnableTensor):
    func: Callable[..., Tensor]
    prepass: PrePass
    args: Tuple[LazyTensor | Tensor | int | float | bool, ...] = dcls.field(
        default_factory=tuple
    )
    kwargs: Dict[str, LazyTensor | Tensor | int | float | bool] = dcls.field(
        default_factory=dict
    )

    def __init__(
        self,
        func: Callable[..., Tensor],
        prepass: PrePass,
        *args: LazyTensor | Tensor | int | float | bool,
        **kwargs: LazyTensor | Tensor | int | float | bool,
    ) -> None:
        self.func = func
        self.prepass = prepass
        self.args = args
        self.kwargs = kwargs

    def __hash__(self) -> int:
        # Evaluations are unique.
        return id(self)

    def _run(self) -> Tensor:
        real_args = [run(arg) for arg in self.args]
        real_kwargs = {k: run(v) for (k, v) in self.kwargs.items()}

        result = self.func(*real_args, **real_kwargs)
        assert self.prepass.shape == result.shape, [self.prepass, result.shape]

        return result

    run = _run

    def visit(self, nodes: Dict[int, TensorLike]) -> None:
        if hash(self) in nodes.keys():
            return

        for arg in self.args:
            if isinstance(arg, Tensor):
                nodes[hash(arg)] = arg
            elif isinstance(arg, RunnableTensor):
                arg.visit(nodes)

        for val in self.kwargs.values():
            if isinstance(val, Tensor):
                nodes[hash(val)] = val
            elif isinstance(val, RunnableTensor):
                val.visit(nodes)

        assert hash(self) not in nodes.keys()
        nodes[hash(self)] = self

    def _take_batch(self, low: int, high: int) -> Tensor:
        logger.debug(
            "Evaluation taking batch: (%s, %s), low=%s, high=%s",
            self.size(),
            self.batch(),
            low,
            high,
        )

        args = [take_batch(arg, low, high) for arg in self.args]
        kwargs = {k: take_batch(v, low, high) for (k, v) in self.kwargs.items()}
        result = self.func(*args, **kwargs)

        if (reducer := self.prepass.reducer()) is None:
            raise UnsupportedError("Cannot safely parallelize.")

        result = reducer(result)
        return result

    take_batch = _take_batch

    def size(self, dim: int | None = None) -> int | Tuple[int, ...]:
        shape = self.prepass.shape
        if dim is not None:
            return shape[dim]
        else:
            return shape

    def dtype(self) -> DType:
        return self.prepass.dtype()

    def device(self) -> str | Device:
        return self.prepass.device()

    def batch(self) -> BatchInfo | None:
        return self.prepass.batch()


@final
@dataclass(init=False, repr=False)
class LazyTensor(RunnableTensor):
    _data: TensorLike
    _batch: BatchInfo | None = None

    def __init__(self, data: TensorLike, batch: int | None = None) -> None:
        if isinstance(data, LazyTensor):
            self._data = data._data
            self._batch = data._batch
        elif isinstance(data, Evaluation):
            self._data = data
            self._batch = data.batch()
        else:
            self._data = data
            if batch is None:
                self._batch = None
            else:
                self._batch = BatchInfo(batch, data.size(batch))

        logger.debug("Creating LazyTensor. %s, %s", type(self._data), self._batch)

    # Implementations

    def run(self) -> Tensor:
        data = self._data
        if isinstance(data, Tensor):
            return data

        return data.run()

    def visit(self, nodes: Dict[int, TensorLike]) -> None:
        data = self._data

        if hash(self) in nodes.keys():
            return

        if isinstance(data, Evaluation):
            data.visit(nodes)
        else:
            nodes[hash(self)] = self

        assert hash(self) in nodes.keys()

    def take_batch(self, low: int, high: int) -> Tensor:
        logger.debug(
            "LazyTensor taking batch: (%s, %s), low=%s, high=%s",
            self.size(),
            self.batch(),
            low,
            high,
        )

        if isinstance(self._data, Tensor):
            if self._batch is None:
                result = self._data
            else:
                result = self._data.index_select(
                    self._batch.index, torch.tensor(list(range(low, high)))
                )
        else:
            result = self._data.take_batch(low, high)

        return result

    @overload
    def size(self) -> Tuple[int, ...]:
        ...

    @overload
    def size(self, dim: int) -> int:
        ...

    def size(self, dim: int | None = None) -> int | Tuple[int, ...]:
        data = self._data

        if dim is None:
            return data.size()

        return data.size(dim)

    def dtype(self) -> DType:
        dt = interfaces.dtyp(self._data)
        return dt

    def device(self) -> str | Device:
        return interfaces.dev(self._data)

    def batch(self) -> BatchInfo | None:
        return self._batch

    # Magic methods

    def __str__(self) -> str:
        return f"LazyTensor {self.run()}"

    def __bool__(self) -> bool:
        return bool(self.item())

    def __int__(self) -> int:
        return int(self.item())

    def __float__(self) -> float:
        return float(self.item())

    def __invert__(self) -> bool:
        return not bool(self)

    def __pos__(self) -> TensorLike:
        return lazy_forward(Tensor.__pos__, prepasses.identity, self)

    def __neg__(self) -> TensorLike:
        return lazy_forward(Tensor.__neg__, prepasses.identity, self)

    def __add__(self, other: TensorLike) -> TensorLike:
        return lazy_forward(Tensor.__add__, prepasses.symmetric, self, other)

    def __radd__(self, other: TensorLike) -> TensorLike:
        return lazy_forward(Tensor.__add__, prepasses.symmetric, other, self)

    def __sub__(self, other: TensorLike) -> TensorLike:
        return lazy_forward(Tensor.__sub__, prepasses.symmetric, self, other)

    def __rsub__(self, other: TensorLike) -> TensorLike:
        return lazy_forward(Tensor.__sub__, prepasses.symmetric, other, self)

    def __mul__(self, other: TensorLike) -> TensorLike:
        return lazy_forward(Tensor.__mul__, prepasses.symmetric, self, other)

    def __rmul__(self, other: TensorLike) -> TensorLike:
        return lazy_forward(Tensor.__mul__, prepasses.symmetric, other, self)

    def __truediv__(self, other: TensorLike) -> TensorLike:
        return lazy_forward(Tensor.__truediv__, prepasses.symmetric, self, other)

    def __rtruediv__(self, other: TensorLike) -> TensorLike:
        return lazy_forward(Tensor.__truediv__, prepasses.symmetric, other, self)

    def __floordiv__(self, other: TensorLike) -> NoReturn:
        del other
        raise UnsupportedError

    def __rfloordiv__(self, other: TensorLike) -> NoReturn:
        del other
        raise UnsupportedError

    def __pow__(self, other: TensorLike) -> TensorLike:
        return lazy_forward(Tensor.__pow__, prepasses.symmetric, self, other)

    def __rpow__(self, other: TensorLike) -> TensorLike:
        return lazy_forward(Tensor.__pow__, prepasses.symmetric, other, self)

    def __mod__(self, other: TensorLike) -> TensorLike:
        return lazy_forward(Tensor.__mod__, prepasses.symmetric, self, other)

    def __rmod__(self, other: TensorLike) -> TensorLike:
        return lazy_forward(Tensor.__mod__, prepasses.symmetric, other, self)

    def __divmod__(self, other: TensorLike) -> NoReturn:
        del other
        raise UnsupportedError

    def __rdivmod__(self, other: TensorLike) -> NoReturn:
        del other
        raise UnsupportedError

    def __abs__(self) -> TensorLike:
        return lazy_forward(Tensor.__abs__, prepasses.identity, self)

    def __hash__(self) -> int:
        # LazyTensors are not unique. They are defined by their data.
        return id(self._data)

    def __matmul__(self, other: TensorLike) -> TensorLike:
        return lazy_forward(Tensor.__matmul__, prepasses.matmul, self, other)

    def __rmatmul__(self, other: TensorLike) -> TensorLike:
        return lazy_forward(Tensor.__matmul__, prepasses.matmul, other, self)

    def __eq__(self, other: TensorLike) -> TensorLike:
        return lazy_forward(Tensor.__eq__, prepasses.symmetric, self, other)

    def __ne__(self, other: TensorLike) -> TensorLike:
        return lazy_forward(Tensor.__ne__, prepasses.symmetric, self, other)

    def __gt__(self, other: TensorLike) -> TensorLike:
        return lazy_forward(Tensor.__gt__, prepasses.symmetric, self, other)

    def __ge__(self, other: TensorLike) -> TensorLike:
        return lazy_forward(Tensor.__ge__, prepasses.symmetric, self, other)

    def __lt__(self, other: TensorLike) -> TensorLike:
        return lazy_forward(Tensor.__lt__, prepasses.symmetric, self, other)

    def __le__(self, other: TensorLike) -> TensorLike:
        return lazy_forward(Tensor.__le__, prepasses.symmetric, self, other)

    def __len__(self) -> int:
        return self.size(0)

    def __getitem__(
        self, index: int | slice | Tensor | List[Any] | Tuple[Any] | None
    ) -> Tensor:
        if isinstance(self._data, RunnableTensor):
            data = self._data.run()
        else:
            data = self._data
        return data[index]

    def __setitem__(
        self,
        index: int | slice | Tensor | List[Any] | Tuple[Any] | None,
        value: Tensor,
    ) -> None:
        if isinstance(self._data, RunnableTensor):
            raise UnsupportedError

        self._data[index] = value

    def __getattr__(self, name: str) -> Callable[..., Any]:
        logger.debug(
            f"__getattr__ called for {name}. Automatically resolving function."
        )

        method = getattr(Tensor, name)
        wrapper = functools.wraps(method)

        if (custom_impl := CUSTOM_OPS.lookup_method(name)) is not None:
            logger.debug("A custom method definition is found.")
            partial = functools.partial(custom_impl, self)
        elif (shape_impl := SHAPE_OPS.lookup_method(name)) is not None:
            logger.debug("A custom shape method is found. Lazy evaluation.")
            partial = functools.partial(lazy_forward, method, shape_impl, self)
        else:
            logger.debug("No custom methods found. Evaluating eagerly.")
            partial = functools.partial(method, run(self))

        return wrapper(partial)

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Tensor],
        types: Tuple[Type[Any], ...],
        args: Sequence[TensorLike] = (),
        kwargs: Dict[str, TensorLike] | None = None,
    ) -> TensorLike:
        if kwargs is None:
            kwargs = {}

        if not builtins.all(
            issubclass(typ, (LazyTensor, Tensor, int, float, bool)) for typ in types
        ):
            return NotImplemented

        name = func.__name__

        if (custom_impl := CUSTOM_OPS.lookup_function(name)) is not None:
            logger.debug("A custom function definition is found.")
            return custom_impl(*args, **kwargs)
        elif (shape_impl := SHAPE_OPS.lookup_function(name)) is not None:
            logger.debug("A custom shape function is found. Lazy evaluation.")
            return lazy_forward(func, shape_impl, *args, **kwargs)
        else:
            logger.debug("No custom method found. Evaluating eagerly.")
            args = [run(arg) for arg in args]
            kwargs = {k: run(v) for (k, v) in kwargs.items()}
            return func(*args, **kwargs)

    @property
    @wraps(Tensor.size)
    def shape(self) -> Tuple[int, ...]:
        return self.size()

    @property
    @wraps(Tensor.dim)
    def ndim(self) -> int:
        return self.dim()

    @property
    @wraps(Tensor.t)
    def T(self) -> TensorLike:
        return self.t()

    def torch(self) -> Tensor:
        return self.run()

    def backward(self) -> None:
        if self._batch is None or not cuda.is_available():
            logger.debug(
                "Unable to parallelize across batches."
                " "
                "Running backward with native pytorch."
            )
            self.run().backward()
        else:
            total = 0
            logger.debug("Able to parallelize across batches. Hooray!")
            for mini_batch_size in gpus.split_batch(
                self.buffer_memory(), self._batch.value
            ):
                logger.debug("Using mini batch size: %d.", mini_batch_size)
                mini_batch = self.take_batch(total, total + mini_batch_size)
                total += mini_batch_size
                mini_batch.backward()


@overload
def lazy(val: Tensor, batch: int | None = None) -> LazyTensor:
    ...


@overload
def lazy(val: LazyTensor) -> LazyTensor:
    ...


@overload
def lazy(val: int) -> int:
    ...


@overload
def lazy(val: float) -> float:
    ...


@overload
def lazy(val: bool) -> bool:
    ...


def lazy(val: Any, batch: int | None = None) -> Any:
    logger.debug("lazy %s, %s", type(val), interfaces.bat(val))

    if isinstance(val, Tensor):
        val = LazyTensor(val, batch)

    return val


@overload
def run(val: LazyTensor) -> Tensor:
    ...


@overload
def run(val: RunnableTensor) -> Tensor:
    ...


@overload
def run(val: Runnable[T]) -> T:
    ...


@overload
def run(val: T) -> T:
    ...


def run(val: Any) -> Any:

    if isinstance(val, Runnable):
        return val.run()

    return val


@overload
def take_batch(tensor: TensorLike, low: int, high: int) -> Tensor:
    ...


@overload
def take_batch(tensor: int, low: int, high: int) -> int:
    ...


@overload
def take_batch(tensor: float, low: int, high: int) -> float:
    ...


@overload
def take_batch(tensor: bool, low: int, high: int) -> bool:
    ...


def take_batch(
    tensor: TensorLike | int | float | bool, low: int, high: int
) -> Tensor | int | float | bool:

    if isinstance(tensor, RunnableTensor):
        logger.debug(
            "Generic taking batch: (%s, %s), low=%s, high=%s",
            tensor.size(),
            tensor.batch(),
            low,
            high,
        )
        return tensor.take_batch(low, high)

    return tensor


def lazy_forward(
    func: Callable[..., Any], shape_func: PrePassFunc, *args: Any, **kwargs: Any
) -> TensorLike:
    if torch.is_grad_enabled():
        out = LazyTensor(LazyFunction(func, shape_func)(*args, **kwargs))
        logger.debug("lazy forward %s, %s", out.size(), out.batch())
        return out
    else:
        return func(*args, **kwargs)


# Functions that require special handling.


class _ValIdx(NamedTuple):
    values: TensorLike
    indices: TensorLike


@overload
def _min(input: TensorLike) -> TensorLike:
    ...


@overload
def _min(input: TensorLike, dim: int, keepdim: bool = False) -> _ValIdx:
    ...


@overload
def _min(input: TensorLike, other: TensorLike) -> TensorLike:
    ...


@wraps(torch.min)
def _min(input: TensorLike, *args: Any, **kwargs: Any) -> TensorLike | _ValIdx:
    if len(args) == len(kwargs) == 0:
        return lazy_forward(torch.min, prepasses.reduce_dims, input)

    if (
        len(args) == 1
        and isinstance((other := args[0]), (Tensor, LazyTensor))
        or len(kwargs) == 1
        and (other := kwargs.get("other", None) is not None)
    ):
        return lazy_forward(torch.minimum, prepasses.symmetric, input, other)

    return _ValIdx(
        lazy_forward(torch.amin, prepasses.reduce_dims, input, *args, **kwargs),
        lazy_forward(torch.argmin, prepasses.reduce_dims, input, *args, **kwargs),
    )


@overload
def _max(input: TensorLike) -> TensorLike:
    ...


@overload
def _max(input: TensorLike, dim: int, keepdim: bool = False) -> _ValIdx:
    ...


@overload
def _max(input: TensorLike, other: TensorLike) -> TensorLike:
    ...


@wraps(torch.max)
def _max(input: TensorLike, *args: Any, **kwargs: Any) -> TensorLike | _ValIdx:
    if len(args) == len(kwargs) == 0:
        return lazy_forward(torch.max, prepasses.reduce_dims, input)

    if (
        len(args) == 1
        and isinstance((other := args[0]), (Tensor, LazyTensor))
        or len(kwargs) == 1
        and (other := kwargs.get("other", None) is not None)
    ):
        return lazy_forward(torch.maximum, prepasses.symmetric, input, other)

    return _ValIdx(
        lazy_forward(torch.amax, prepasses.reduce_dims, input, *args, **kwargs),
        lazy_forward(torch.argmax, prepasses.reduce_dims, input, *args, **kwargs),
    )


def _permute_function_shape(
    input: TensorLike, dims: int | Tuple[int, ...], *args: Any, **kwargs: Any
) -> PrePass:
    prepasses.mute_unused_args(*args, **kwargs)

    if isinstance(dims, int):
        dims = (dims,)

    return prepasses.permute(input, *dims)


def _reshape_function_shape(
    input: TensorLike, dims: Tuple[int, ...], *args: Any, **kwargs: Any
) -> PrePass:
    prepasses.mute_unused_args(*args, **kwargs)

    return prepasses.reshape(input, *dims)


def _t_shape(input: TensorLike, *args: Any, **kwargs: Any) -> PrePass:
    prepasses.mute_unused_args(*args, **kwargs)

    return prepasses.tranpose(input, 0, 1)


@dataclass
class MethodFunction(Generic[T]):
    method: Dict[str, T]
    function: Dict[str, T]

    @staticmethod
    def _search(key: str, *dbs: Dict[str, T]) -> T | None:
        for db in dbs:
            if (value := db.get(key)) is not None:
                return value
        return None

    def lookup(self, key: str, *dbs: Dict[str, T]) -> T | None:
        if (result := self._search(key, *dbs)) is not None:
            return result

        if key.startswith("_"):
            fallback = key.lstrip("_")
            return self._search(fallback, *dbs)
        return None

    def lookup_method(self, key: str) -> T | None:
        return self.lookup(key, self.method, self.function)

    def lookup_function(self, key: str) -> T | None:
        return self.lookup(key, self.function)


CUSTOM_OPS = MethodFunction[Callable](
    method={},
    function={
        "min": _min,
        "max": _max,
    },
)

PARTIAL_OPS = MethodFunction[Callable](method={}, function={"sum": lambda x: x})

SHAPE_OPS = MethodFunction[PrePassFunc](
    method={"permute": prepasses.permute, "view": prepasses.view},
    function={
        "positive": prepasses.identity,
        "negative": prepasses.identity,
        "neg": prepasses.identity,
        "add": prepasses.symmetric,
        "sub": prepasses.symmetric,
        "subtract": prepasses.symmetric,
        "mul": prepasses.symmetric,
        "multiply": prepasses.symmetric,
        "div": prepasses.symmetric,
        "divide": prepasses.symmetric,
        "true_divide": prepasses.symmetric,
        "floor": prepasses.identity,
        "fmod": prepasses.symmetric,
        "remainder": prepasses.symmetric,
        "frac": prepasses.identity,
        "pow": prepasses.symmetric,
        "exp": prepasses.identity,
        "exp2": prepasses.identity,
        "log": prepasses.identity,
        "log2": prepasses.identity,
        "log10": prepasses.identity,
        "log1p": prepasses.identity,
        "abs": prepasses.identity,
        "matmul": prepasses.matmul,
        "bmm": prepasses.matmul,
        "mm": prepasses.matmul,
        "mv": prepasses.matmul,
        "dot": prepasses.matmul,
        "eq": prepasses.symmetric,
        "equal": prepasses.symmetric,
        "ne": prepasses.symmetric,
        "not_equal": prepasses.symmetric,
        "gt": prepasses.symmetric,
        "greater": prepasses.symmetric,
        "ge": prepasses.symmetric,
        "greater_equal": prepasses.symmetric,
        "lt": prepasses.symmetric,
        "less": prepasses.symmetric,
        "le": prepasses.symmetric,
        "less_equal": prepasses.symmetric,
        "mean": prepasses.mean,
        "sum": prepasses.reduce_dims,
        "std": prepasses.reduce_dims,
        "minimum": prepasses.symmetric,
        "maximum": prepasses.symmetric,
        "amin": prepasses.reduce_dims,
        "amax": prepasses.reduce_dims,
        "argmin": prepasses.reduce_dims,
        "argmax": prepasses.reduce_dims,
        "isclose": prepasses.symmetric,
        "cat": prepasses.cat,
        "t": _t_shape,
        "permute": _permute_function_shape,
        "reshape": _reshape_function_shape,
        "flatten": prepasses.flatten,
        "transpose": prepasses.tranpose,
        "select": prepasses.select,
        "index_select": prepasses.select,
        "sin": prepasses.identity,
        "cos": prepasses.identity,
        "tan": prepasses.identity,
        "asin": prepasses.identity,
        "acos": prepasses.identity,
        "atan": prepasses.identity,
        "sinh": prepasses.identity,
        "cosh": prepasses.identity,
        "tanh": prepasses.identity,
        "asinh": prepasses.identity,
        "acosh": prepasses.identity,
        "atanh": prepasses.identity,
        "sigmoid": prepasses.identity,
        "hardsigmoid": prepasses.identity,
        "softmax": prepasses.identity,
        "relu": prepasses.identity,
        "relu6": prepasses.identity,
        "leaky_relu": prepasses.identity,
        "l1_loss": prepasses.scalars,
        "mse_loss": prepasses.scalars,
        "cross_entropy": prepasses.scalars,
        "binary_cross_entropy": prepasses.scalars,
        "binary_cross_entropy_with_logits": prepasses.scalars,
        "elu": prepasses.identity,
        "gelu": prepasses.identity,
        "dropout": prepasses.identity,
        "batch_norm": prepasses.identity,
        "layer_norm": prepasses.identity,
        "linear": prepasses.linear,
        "embedding": prepasses.embedding,
        "pad": prepasses.pad,
        "conv1d": prepasses.conv,
        "conv2d": prepasses.conv,
        "conv3d": prepasses.conv,
        "conv_transpose1d": prepasses.conv_transpose,
        "conv_transpose2d": prepasses.conv_transpose,
        "conv_transpose3d": prepasses.conv_transpose,
        "max_pool1d": prepasses.maxpool,
        "max_pool2d": prepasses.maxpool,
        "max_pool3d": prepasses.maxpool,
        "avg_pool1d": prepasses.avgpool,
        "avg_pool2d": prepasses.avgpool,
        "avg_pool3d": prepasses.avgpool,
        # Functions that will not be implemented.
        "__floordiv__": UnsupportedError.raise_error,
    },
)
