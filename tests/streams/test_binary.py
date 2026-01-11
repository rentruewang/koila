# Copyright (c) AIoWay Authors - All Rights Reserved

from collections import Counter
from collections.abc import Callable

import pytest
import tensordict
import torch
from pytest import FixtureRequest
from tensordict import TensorDict

from aioway.streams import (
    CacheStream,
    ListStream,
    NestedLoopJoinStream,
    Stream,
    ZipStream,
)


@pytest.fixture
def lhs_stream(concat_stream: Stream) -> CacheStream:
    return CacheStream(concat_stream)


@pytest.fixture
def rhs_stream(joinable_stream: Stream) -> CacheStream:
    return CacheStream(joinable_stream)


def test_lhs_stream_length(concat_stream: Stream, lhs_stream: Stream):
    assert concat_stream.size == lhs_stream.size


def test_rhs_stream_length(joinable_stream: Stream, rhs_stream: Stream):
    assert joinable_stream.size == rhs_stream.size


@pytest.fixture
def binary_stream(
    request: FixtureRequest,
    lhs_stream: CacheStream,
    rhs_stream: CacheStream,
) -> Stream:
    "An indirect fixture that takes in a builder function and outputs a stream."

    builder: Callable[[Stream, Stream], Stream] = request.param

    if not callable(builder):
        raise TypeError("Indirect fixture `binary_stream` only accepts functions.")

    return builder(lhs_stream, rhs_stream)


def _zip_builder(lhs_stream: CacheStream, rhs_stream: CacheStream) -> ZipStream:
    return ZipStream(left=lhs_stream, right=rhs_stream)


@pytest.mark.parametrize("binary_stream", [_zip_builder], indirect=True)
def test_zip_input_len(
    binary_stream: ZipStream, concat_stream: Stream, rhs_stream: Stream
):
    assert min(concat_stream.size, rhs_stream.size) == binary_stream.size


@pytest.mark.parametrize("binary_stream", [_zip_builder], indirect=True)
def test_zip(
    binary_stream: ZipStream,
    lhs_stream: CacheStream,
    rhs_stream: CacheStream,
):
    assert not lhs_stream.started
    assert not rhs_stream.started
    assert not binary_stream.started
    assert binary_stream.left is lhs_stream
    assert binary_stream.right is rhs_stream
    for result in binary_stream:
        assert binary_stream.idx == lhs_stream.idx == rhs_stream.idx
        concat = tensordict.merge_tensordicts(
            lhs_stream[lhs_stream.idx - 1], rhs_stream[rhs_stream.idx - 1]
        )
        assert (result == concat).all()


def _join_builder(
    lhs_stream: CacheStream, rhs_stream: CacheStream
) -> NestedLoopJoinStream:
    return NestedLoopJoinStream(left=lhs_stream, right=rhs_stream, key="i1d")


@pytest.mark.parametrize("binary_stream", [_join_builder], indirect=True)
def test_join_input_len(
    binary_stream: NestedLoopJoinStream,
    lhs_stream: CacheStream,
    rhs_stream: CacheStream,
):
    assert binary_stream.size == lhs_stream.size * rhs_stream.size


@pytest.mark.parametrize(
    "to_slice",
    [
        lambda x: [x],
        lambda t: [t[0:2], t[2:4]],
        lambda t: [t[[1, 3]], t[[0, 2]]],
    ],
)
def test_simple_nested_loop_join(to_slice: Callable[[TensorDict], list[TensorDict]]):
    left = TensorDict({"a": [1, 3, 2, 2], "b": [4, 10, 5, 6]}, batch_size=4)
    right = TensorDict({"a": [1, 3, 2, 2], "c": [7, 11, 8, 9]}, batch_size=4)

    left_stream = ListStream(to_slice(left))
    right_stream = ListStream(to_slice(right))

    out = tensordict.cat(
        list(
            NestedLoopJoinStream(
                left_stream,
                CacheStream(right_stream),
                key="a",
            )
        )
    )

    def sort_by_abc(td: TensorDict):
        for key in "cba":
            indices = torch.argsort(td[key], stable=True)
            td = td[indices]
        return td

    assert (
        sort_by_abc(out)
        == {
            "a": [1, 2, 2, 2, 2, 3],
            "b": [4, 5, 5, 6, 6, 10],
            "c": [7, 8, 9, 8, 9, 11],
        }
    ).all()


@pytest.mark.xfail
@pytest.mark.parametrize("binary_stream", [_join_builder], indirect=True)
def test_join_equal_as_original(
    binary_stream: NestedLoopJoinStream,
    lhs_stream: Stream,
    rhs_stream: CacheStream,
):
    block_frame_block = tensordict.cat(list(lhs_stream))
    joinable_frame_block = tensordict.cat(list(rhs_stream))

    # Performing the join here.
    results: list[TensorDict] = list(binary_stream)
    assert len(results), "The binary stream is empty."
    answer_items = tensordict.cat(results)["i1d"]

    # Do it at once, using ``ListStream`` as it yields everything in 1 batch.
    ground_truth = tensordict.cat(
        list(
            NestedLoopJoinStream(
                left=ListStream([block_frame_block]),
                right=CacheStream(ListStream([joinable_frame_block])),
                key="i1d",
            )
        )
    )

    answer_count = Counter(answer_items.tolist())
    truth_count = Counter(ground_truth["i1d"].tolist())

    assert answer_count == truth_count


@pytest.mark.xfail
@pytest.mark.parametrize("binary_stream", [_join_builder], indirect=True)
def test_match_functionally(
    binary_stream: NestedLoopJoinStream,
    lhs_stream: CacheStream,
    rhs_stream: CacheStream,
):
    block_frame_block = tensordict.cat(list(lhs_stream))
    joinable_frame_block = tensordict.cat(list(rhs_stream))

    # Performing the join here.
    results: list[TensorDict] = list(binary_stream)
    answer_items = tensordict.cat(results)["i1d"]

    answer_count = Counter(answer_items.tolist())

    left_count = Counter(block_frame_block["i1d"].tolist())
    right_count = Counter(joinable_frame_block["i1d"].tolist())

    # Functionally correct join.
    assert left_count.keys() == {*block_frame_block["i1d"].tolist()}
    assert right_count.keys() == {*joinable_frame_block["i1d"].tolist()}
    assert answer_count.keys() == left_count.keys() & right_count.keys()

    for key in answer_count.keys():
        assert answer_count[key] == left_count[key] * right_count[key]


@pytest.mark.parametrize(
    "binary_stream",
    [_zip_builder, _join_builder],
    indirect=True,
)
def test_binary_stream_in_list(binary_stream: NestedLoopJoinStream | ZipStream):
    assert binary_stream.size, "The binary stream is empty."

    assert binary_stream.idx == 0, "Pre iteration stream's index starts with 0."

    batches: list[TensorDict] = []
    for idx, batch in enumerate(binary_stream, start=1):
        # Ensure that the input is also exhausted.
        assert idx == binary_stream.idx
        batches.append(batch)

    assert binary_stream.idx == binary_stream.size == len(batches)
