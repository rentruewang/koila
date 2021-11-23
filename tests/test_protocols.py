import koila
from koila import Lazy, Runnable


def test_is_runnable() -> None:
    class MyCustomRunnable:
        num_elements = 0

        def run(self) -> int:
            return 42

    mcr = MyCustomRunnable()
    assert isinstance(mcr, Runnable)

    assert koila.run(mcr) == 42


def test_lazy_is_runnable() -> None:
    assert issubclass(Lazy, Runnable)
