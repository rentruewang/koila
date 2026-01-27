# Datasets


A dataset is a collection of `Block`s. Here are the main types of datasets:

1. `Table`: A bounded
2. `Stream`: An unbounded stream of `Block`s.

Where tables can make use of `Index` to make indexing faster.

Dependency wise, datasets are a layer on top of `Block`s, making use of its features.
