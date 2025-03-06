Physical layers
###############

Eventually, the logical plan would need to be converted to a physical plan,
and along with data being passed in, execute the plan.

In ``aioway``, physical plans are

``Frame``s
**********

``Frame`` store data, similar to ``Dataset`` from ``torch``.

It appears to be a sequence of ``Block``s, representing a possibly distrubted data storage.

Originally, ``aioway``'s physical operations are conducted using ``Frame``s,
however, the streaming nature of machine learning is incompatible with random accessing,
because each ``__getitem__`` call computes the ``Block`` on the fly,
and random access makes caching difficult.

Therefore, ``Frame`` eventually became just a cache,
and would be used in computation when we are doing computation in a staged-based manner.

``Stream``s
***********

``Stream`` is an ``Iterator`` over existing data, possibly performing some computation.

``Stream`` is the most natural representation during model training and inference,
as well as for relational algebra. It's tree-like structure also allows joint optimization,
in addition to allow tracing of the full computation graph, so it's desirable.


``Block``s
**********

``Block``s are used to represent batches, currently a thin wrapper over ``TensorDict``,
a batch of data that can move around different devices.

In the future, this might be generalized for models
that use the different length for the same inputs, such as ``torch_geometric``.

``Index``s
**********

``Index``s are responsible for providing very fast access by index lookup.
However, since it cannot handle data being dynamically added in,
it can only be added on ``Frame``s, but not ``Stream``s.

There are several different kinds of ``Index``s, including but not limited to:

#. Vector search index.
   This is the kinds of indices that does approximate nearest neighbor search,
   and the results would yield both distances and indices of the nearest neighbors.

#. Primitive value index.
   This includes strings, integers, and floating point numbers,
   this can be implemented as a key-value store,
   which would be ``dict``s in memory, or No-SQL DBs like ``redis`` externally.

#. Range index.
   In a distributed setting, this is used for random accessing a ``Frame``.

#. Others.
   Maybe draw inspiration from neo4j indices? Text search indices, point indices etc.
   Those can occasionally be helpful.

``Table``s
**********

``Table``s are the manager of the physical plan.
It is responsible for gluing together ``Frame`` and ``Index``,
and produce ``Stream`` while supplying the runtime behavior.
