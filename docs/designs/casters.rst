Casting between types
#####################

Different data has different data types,
which is central to how ``aioway`` works under the hood.

Often times data structures between different implementations don't work well together,
either cannot be used together (``numpy`` and ``torch``),
or doesn't have a compatible API (``pandas`` and ``numpy``).

This simply means that we would need to do our own casting and providing a unified business logic API.

So, how do we do it?

Efficiency vs complexity
************************

If we define different subtypes of our phsyical abstraction, a neccessity to avoid translation costs,
we would need multiple implementations over the same class (for example, ``Block``).

However, this comes at the cost of complexity, even mentally,
because we need to define all the methods converting to different types on all classes.

However, if we draw inspiration from ``numpy``'s ``astype`` method,
which allows users to convert between different data types,
while only needing to provide 1 global matrix (type conversion matrix).

For this purpose, I have written a type called ``Caster``,
providing the convenient ``astype`` methods with a standardized interface.

Or maybe deal with the translation cost
***************************************

Internally, whenever we are translating from ``numpy`` to ``torch``, there is a cost.

If we are converting from ``ibis`` to ``torch``,
the cost is much larger because we would need to pull everything into memory, uncompressed.

However, during the prototype phase, the cost may be negligible,
compared to the multiple class changes we have to do everytime someone updates the API.

I think for now, let's simplify the costs first and then later add them back.
