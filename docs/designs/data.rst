Shared and temporary data
#########################

Shared and temporary data refer to the data that we have available,
that is deserialized from the backends,
in order to perform some analysis or for transfering between backends.

Persistent storage
******************

The main consideration of this temporary data storage is whether or not to make it persistent.
Aioway at its core is a streaming engine.
That is to say, there isn't a need for aioway to persist data on its own.
However, since it is required to access data sources accross different frameworks,
using a nosql database like redis might be a good way to approach this problem,
and it has the benefit of being very scalable.

``Block``s vs ``Table``s
************************

The main physical abstractions in this project, ``Block``s and ``Table``s, are very similar.
Both of them have a ``DataFrame``-like API, which supports relational algebra.
The main difference comes in the fact that ``Block`` is our in-memory data structure,
similar to how a ``pandas.DataFrame`` operates, while ``Table`` is a producer of ``Block``,
can possibly be lazily evaluated, and preserves relational information.
