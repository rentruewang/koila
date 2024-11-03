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
