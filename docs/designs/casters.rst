Casting between types
#####################

Different data has different data types,
which is central to how ``aioway`` works under the hood.

Often times data structures between different implementations don't work well together,
either cannot be used together (``numpy`` and ``torch``),
or doesn't have a compatible API (``pandas`` and ``numpy``).

This simply means that we would need to do our own casting and providing a unified business logic API.

So, how do we do it?

Assumption
**********

When we are using a framework,
we make the assumption that using operations a framework supports natively should be the fastest.

This is to say, having no external operation (like serializing and deserializing) is faster.

For example, using the different list interfaces as examples;
if we have a family of classes e.g. array list and linked list,
normally we can convert them by hand or just transfer data in between “normally”.

However, doing so means serializing and deserializing from device if the data is not locally on the same machine,
or if serializing and deserializing is expensive e.g. numpy or torch,
and this is where auto casting can help.
This way numpy / torch can auto cast to the desired dtype.

In this case, the frameworks are the classes (numpy array),
and non native operations like loading into python slows it down.

By doing auto casting, we optimize away the part where we would need to have to cast things manually.

Primitive types also has this.
And this should be completely hidden from the user.
Otherwise, user casting has more benefit.

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
