# AioWay

## 🚧 Notice

Aioway is a work in progress, builds on top of the original [`koila` (moved to a branch)][koila]. The `torch` team built `FakeTensor` which overlaps a lot with `koila`'s functionality, so it's no longer maintained. See the rationale in the [koila][koila] branch.

Conceptually, `aioway` works in a similar way, but instead of `Tensor` ops, `aioway` focuses on a higher level, on **algorithm building**. See below for the promised features:

## 🍰 Promised features

- Simple and declarative.
- Detects the tasks at hand, resource available, and select the best algorithms and models.
- The models built from aioway would be white box (explanable).
- Allows you to scale up the model size, and to different machines.
- Extensible with custom pytorch.

## 🗺️ Roadmap

For the pre-release version (`v0.0.*`), see [project](https://github.com/users/rentruewang/projects/7) for more details.


## 🤔 Why aioway yada yada

In the recent years, machine learning's entry barrier has gotten higher, rather than lower. With the increasing number of algorithms and libraries and models, it's no wonder qualified data scientists are rare because you would need years of training to keep up to the status quo.

We designed Aioway in a way such that instead of thinking about **how** to do ML, you specify **what** to do. Instead of focusing on what algorithms and models to use, Aioway allows you to focus on the use cases by taking into account the context of the problem, and perform compliation according to the data to ensure good performance. Automatically.

[koila]: https://github.com/rentruewang/aioway/tree/koila
