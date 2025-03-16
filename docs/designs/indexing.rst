Indexing
########

Indexes are a way to perform searching with sub-linear time complexity
by organizing data storage in a clever way,
or builds a smart mapping with a special structuer
to reduce the amount of data needed to traverse in order to find the desired match.

Usage
*****

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
