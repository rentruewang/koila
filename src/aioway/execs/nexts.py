# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing
from collections.abc import Callable

from aioway.blocks import Block
from aioway.errors import AiowayError

if typing.TYPE_CHECKING:
    from .execs import Exec

__all__ = ["ExecNextMethod"]


@dcls.dataclass(frozen=True)
class ExecNextMethod:
    """
    Compute the next block of data for ``Exec``.

    Note:
        Since ``Exec``s are a poll system of ``Iterator``s, those iterators must be in sync.
        However, having a DAG of ``Exec``s means that some of the computation is shared,
        which is extremely common place.

        However, if we follow a naive ``__next__`` scheme, this means that
        not all of the ``Exec``s are only called once, which can lead to bugs.

        Assuming that ``Exec``s are public API, we must choose from one of the cases:

        #. The computation is not shared,
            and the same computation is performed multiple times.
        #. The computation is shared, by means of some buffer.
            This needs to be special cased and is not scalable.
        #. The computation is shared with a new ``Exec``, say ``BufferExec``.
            However, this ``BufferExec`` is not a good abstraction,
            because calling ``next`` 2 times should give the same result
            until buffer is cleared, which violates basic assumption of `next`.

        None of which is worth it.

        However, if ``Exec``s are not public API, we can manage the ``Iterator``s ourselves,
        which means that we can make use of the following patterns:

        #. Call ``next`` carefully.
        #. Use contexts to manage the execution.
        #. Use decorators to manage the execution.

    Note:
        ``ExecNextMethod`` takes the approach of implicit doubly recursion, detailed below.

        Because we want to perform some common functionalities (e.g. tracing, shortcutting)
        during the recursive polling of the executors,
        there are multiple designs each with their own trade-offs.
        Eventually, I have settled on this doubly-recursive call design.

        #. Doubly recursion

            This is the classical pattern, implemented in e.g. spark, postgres,
            where a recursive function would call a (final) public function,
            which would call a private method that should be overridden.

            Since we can store states in subclass objects,
            this approach is very flexible,
            as it can represent different arguments with the same interface.

            However, the downside to this approach is that it's not very easy to parse,
            and not too flexible in rewriting the graph, as you need to rewrite everything,
            or make the nodes in the graph mutable, as perhaps the interface is too flexible.

        #. Interpreters

            Used by e.g. clickhouse (for now, before in 2016 they used doubly recursion),
            where subclass defines a declarative function,
            and submits the execution into an interpreter.

            This approach is efficient and easy to optimize,
            but difficult to design properly,
            as a lot of special casing would be needed in the interpreters,
            to deal with different (imcompatible) types of functions.

        #. Implicit doubly recursion

            Still doubly recursion, except we write as a simple recursion,
            but make use of decorators to bring in additional contexts.

            Advantage is that this approach is very elegant in writing.
            However, you have to store the object in addition to the overwritten method,
            and have to be sure that you have access to both ``self`` and the original method,
            making it potentially more complicated.
    """

    impl: Callable[["Exec"], Block]

    def __get__(self, instance: "Exec | None", owner: type["Exec"]):
        """
        Returns a bounded method if the instance is not None,
        otherwise returns the unbound method.
        """

        if instance is None:
            return self

        # If the instance is not None, we are in a bound method.
        # We need to return a new function that calls the original function
        # with the instance as the first argument.
        return lambda *args, **kwargs: self.impl(instance, *args, **kwargs)


class ExecNextMethodUnboundError(AiowayError, AttributeError): ...
