# `TensorFn`s and `TensorDictFn`s.

`TensorFn` are `Fn`s (delayed computation)'s `Tensor` version.

In addition to `do()` and `deps()` functions that `Fn`s support,
`TensorFn` should additionally define the `preview()` function
that generates a quick preview of the current `TensorFn`.
