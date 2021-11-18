from koila import LazyTensor
from koila.protocols import Runnable


def test_tensor_is_runnable() -> None:
    assert issubclass(LazyTensor, Runnable)
