# Copyright (c) AIoWay Authors - All Rights Reserved

"""
Indexing
########

Indexes are a way to perform searching with sub-linear time complexity
by organizing data storage in a clever way,
or builds a smart mapping with a special structuer
to reduce the amount of data needed to traverse in order to find the desired match.

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

Usage
*****

Index work on ``Frame``s because their size / batch count is known beforehand.

Building an index only makes sense if an index is going to be used multiple times in the future,
because building and index takes as long as traversing it + computing,
and you also need to pay the price for storage.

Materialization
***************

Since building an index requires traversing over the input data,
as well as caching it into another representation,
it effectively requires materialization to be performed on the input data.

This can be a strong constraint for streaming backends, and needed to be handled carefully.
This means that building and index either has to be incremental,
or does not occur at all during inference, for instance.

For now, indexing on ``Frame``s is allowed,
while on ``Stream`` it is not, because ``Frame``s are materialized and stored.

todo))
   Use this compoent, test this component.
"""

from .faiss import *
from .indices import *
from .mgrs import *
from .ops import *
