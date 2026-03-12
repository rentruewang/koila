# Batches

Batches represent data that is in memory and processed together.

## `Chunk`s

`Chunk`s represent a block of heterogenious data,
backed by a `TensorDict` and `AttrSet`.
It performs data validation automatically on initialization.

It is currently an immutable data structure,
to make it easier to implement and reason,
at the cost of being harder to use.
This can change in the future.

## `Vector`s

`Vector`s are a value in a `Chunk`. It contains homogenious data.
It is backed by a `Tensor`.
