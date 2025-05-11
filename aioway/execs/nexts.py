# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import typing
from collections.abc import Callable

from aioway.blocks import Block
from aioway.errors import AiowayError

if typing.TYPE_CHECKING:
    from .execs import Exec


@dcls.dataclass(frozen=True)
class ExecNextMethod:
    """
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

    def __get__(self, instance: "Exec" | None, owner: type["Exec"]):
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
