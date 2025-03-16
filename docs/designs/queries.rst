Query optimizers
################

It is possible to leverage existing query optimizers to perform optimization,
because this is essentially based on relational algebra, much like catalyst for spark.

However, as of right now, I am not planning to do it yet,
because there are more things to do before this.

Query optimization
******************

After the logical plans are given,
query optimization can help in simplifying the plans according to some algorithms and heuristics.

This might have some overlaps with `cost-based optimization`_, see the section below.

Cost-based optimization
***********************

Cost based optimization transforms the given logical plan into a physical plan based on some defined costs.
The costs are related to the physical operators and how expensive they are given an underlying architecture.

The original plans are for me to also implement this, similar to spark.
However, I have seen spark's progress, it takes a long time to solve this,
so let's not do this at this moment because I have no time for it.
Let's delegate this to the backends for now, of which spark is one.
