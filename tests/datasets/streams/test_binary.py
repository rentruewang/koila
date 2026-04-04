# Copyright (c) AIoWay Authors - All Rights Reserved

import collections
from collections import abc as cabc

import pytest
import torch

from aioway import chunks, datasets, meta


@pytest.fixture
def lhs_stream(concat_stream: datasets.Stream) -> datasets.CacheStream:
    return datasets.CacheStream(concat_stream)


@pytest.fixture
def rhs_stream(joinable_stream: datasets.Stream) -> datasets.CacheStream:
    return datasets.CacheStream(joinable_stream)


def test_lhs_stream_length(concat_stream: datasets.Stream, lhs_stream: datasets.Stream):
    assert concat_stream.size == lhs_stream.size


def test_rhs_stream_length(
    joinable_stream: datasets.Stream, rhs_stream: datasets.CacheStream
):
    assert joinable_stream.size == rhs_stream.size


@pytest.fixture
def binary_stream(
    request: pytest.FixtureRequest,
    lhs_stream: datasets.Stream,
    rhs_stream: datasets.CacheStream,
):
    "An indirect fixture that takes in a builder function and outputs a stream."

    builder: cabc.Callable[[datasets.Stream, datasets.Stream], datasets.Stream] = (
        request.param
    )

    if not callable(builder):
        raise TypeError("Indirect fixture `binary_stream` only accepts functions.")

    result = builder(lhs_stream, rhs_stream)
    assert isinstance(result, datasets.Stream)
    return result


def _zip_builder(lhs_stream: datasets.Stream, rhs_stream: datasets.CacheStream):
    return datasets.ZipStream(left=lhs_stream, right=rhs_stream)


@pytest.mark.parametrize("binary_stream", [_zip_builder], indirect=True)
def test_zip_input_len(
    binary_stream: datasets.Stream,
    concat_stream: datasets.Stream,
    rhs_stream: datasets.CacheStream,
):
    assert min(concat_stream.size, rhs_stream.size) == binary_stream.size


@pytest.mark.parametrize("binary_stream", [_zip_builder], indirect=True)
def test_zip(
    binary_stream: datasets.Stream,
    lhs_stream: datasets.Stream,
    rhs_stream: datasets.Stream,
):
    assert not lhs_stream.started
    assert not rhs_stream.started
    assert not binary_stream.started
    assert binary_stream.left is lhs_stream
    assert binary_stream.right is rhs_stream
    for result in binary_stream:
        assert binary_stream.idx == lhs_stream.idx == rhs_stream.idx
        concat = lhs_stream[lhs_stream.idx - 1].zip(rhs_stream[rhs_stream.idx - 1])
        assert result == concat


def _join_builder(lhs_stream: datasets.Stream, rhs_stream: datasets.CacheStream):
    return datasets.NestedLoopJoinStream(left=lhs_stream, right=rhs_stream, key="i1d")


@pytest.mark.parametrize("binary_stream", [_join_builder], indirect=True)
def test_join_input_len(
    binary_stream: datasets.Stream,
    lhs_stream: datasets.Stream,
    rhs_stream: datasets.CacheStream,
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
def test_simple_nested_loop_join(
    to_slice: cabc.Callable[[chunks.Chunk], list[chunks.Chunk]],
):
    left = chunks.Chunk.from_data_schema(
        data={"a": torch.tensor([1, 3, 2, 2]), "b": torch.tensor([4, 10, 5, 6])},
        schema=dict(
            a=meta.Attr.parse(
                device="cpu",
                dtype="int64",
                shape=[1],
            ),
            b=meta.Attr.parse(
                device="cpu",
                dtype="int64",
                shape=[1],
            ),
        ),
    )
    right = chunks.Chunk.from_data_schema(
        data={"a": torch.tensor([1, 3, 2, 2]), "c": torch.tensor([7, 11, 8, 9])},
        schema=dict(
            a=meta.Attr.parse(
                device="cpu",
                dtype="int64",
                shape=[1],
            ),
            c=meta.Attr.parse(
                device="cpu",
                dtype="int64",
                shape=[1],
            ),
        ),
    )

    left_stream = datasets.ListStream(to_slice(left))
    right_stream = datasets.ListStream(to_slice(right))

    out = chunks.Chunk.cat(
        list(
            datasets.NestedLoopJoinStream(
                left_stream,
                datasets.CacheStream(right_stream),
                key="a",
            )
        )
    )

    def sort_by_abc(td: chunks.Chunk):
        for key in "cba":
            indices = torch.argsort(td[key].torch(), stable=True)
            td = td[indices]
        return td

    assert sort_by_abc(out) == {
        "a": [1, 2, 2, 2, 2, 3],
        "b": [4, 5, 5, 6, 6, 10],
        "c": [7, 8, 9, 8, 9, 11],
    }


@pytest.mark.parametrize("binary_stream", [_join_builder], indirect=True)
def test_join_equal_as_original(
    binary_stream: datasets.Stream,
    lhs_stream: datasets.Stream,
    rhs_stream: datasets.CacheStream,
):
    block_frame_block = chunks.Chunk.cat(list(lhs_stream))
    joinable_frame_block = chunks.Chunk.cat(list(rhs_stream))

    # Performing the join here.
    results: list[chunks.Chunk] = list(binary_stream)
    assert len(results), "The binary stream is empty."
    answer_items = chunks.Chunk.cat(results)["i1d"]

    # Do it at once, using `datasets.ListStream` as it yields everything in 1 batch.
    ground_truth = chunks.Chunk.cat(
        list(
            datasets.NestedLoopJoinStream(
                left=datasets.ListStream([block_frame_block]),
                right=datasets.CacheStream(datasets.ListStream([joinable_frame_block])),
                key="i1d",
            )
        )
    )

    answer_count = collections.Counter(answer_items.tolist())
    truth_count = collections.Counter(ground_truth["i1d"].tolist())

    assert answer_count == truth_count


@pytest.mark.parametrize("binary_stream", [_join_builder], indirect=True)
def test_match_functionally(
    binary_stream: datasets.Stream,
    lhs_stream: datasets.Stream,
    rhs_stream: datasets.CacheStream,
):
    block_frame_block = chunks.Chunk.cat(list(lhs_stream))
    joinable_frame_block = chunks.Chunk.cat(list(rhs_stream))

    # Performing the join here.
    results = list(binary_stream)
    answer_items = chunks.Chunk.cat(results)["i1d"]

    answer_count = collections.Counter(answer_items.tolist())

    left_count = collections.Counter(block_frame_block["i1d"].tolist())
    right_count = collections.Counter(joinable_frame_block["i1d"].tolist())

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
def test_binary_stream_in_list(
    binary_stream: datasets.NestedLoopJoinStream | datasets.ZipStream,
):
    assert binary_stream.size, "The binary stream is empty."

    assert binary_stream.idx == 0, "Pre iteration stream's index starts with 0."

    batches: list[chunks.Chunk] = []
    for idx, batch in enumerate(binary_stream, start=1):
        # Ensure that the input is also exhausted.
        assert idx == binary_stream.idx
        batches.append(batch)

    assert binary_stream.idx == binary_stream.size == len(batches)
