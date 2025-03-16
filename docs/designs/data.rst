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


Planned backends
****************

There are 2 currently planned backend:

Local / Torch distributed
=========================

This is the default backend. Data are organized into ``Frame``s and ``Stream``s

Materialization can be very difficult because the torch backend assumes that everything is in minibatch.
This means that materialization requires syncrhonization, and cannot be performed liberally,
because CUDA: OOM is a frequent occurence.

Therefore, some careful designs might be needed.

Spark
=====

Materialization does not affect the spark backend all too much,
considering that spark's RDD API is distributed, not streaming.

It might pose a problem for spark structured streaming, though.
