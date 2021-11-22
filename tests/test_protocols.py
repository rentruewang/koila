import koila
from koila import Runnable


def test_is_runnable() -> None:
    class MyCustomRunnable:
        def run(self) -> int:
            return 42

    mcr = MyCustomRunnable()
    assert isinstance(mcr, Runnable)

    assert koila.run(mcr) == 42
