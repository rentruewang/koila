# Copyright (c) AIoWay Authors - All Rights Reserved

from collections import Counter
from collections.abc import Callable

import pytest
import tensordict
from pytest import FixtureRequest
from tensordict import TensorDict

from aioway.streams import (
    NestedLoopJoinStream,
    Stream,
    ZipStream,
)
from aioway.tables import TableStream, TensorDictListTable


@pytest.fixture
def lhs_stream(concat_stream: Stream) -> TableStream:
    return TensorDictListTable.consume(concat_stream).stream()


@pytest.fixture
def rhs_stream(joinable_stream: Stream) -> TableStream:
    return TensorDictListTable.consume(joinable_stream).stream()


@pytest.fixture
def binary_stream(
    request: FixtureRequest,
    lhs_stream: TableStream,
    rhs_stream: TableStream,
) -> Stream:
    "An indirect fixture that takes in a builder function and outputs a stream."

    builder: Callable[[Stream, Stream], Stream] = request.param

    if not callable(builder):
        raise TypeError("Indirect fixture `binary_stream` only accepts functions.")

    return builder(lhs_stream, rhs_stream)


def _zip_builder(lhs_stream: TableStream, rhs_stream: TableStream) -> ZipStream:
    return ZipStream(left=lhs_stream, right=rhs_stream)


@pytest.mark.parametrize("binary_stream", [_zip_builder], indirect=True)
def test_zip_input_len(
    binary_stream: ZipStream, concat_stream: Stream, rhs_stream: Stream
):
    assert len(concat_stream) == len(rhs_stream) == len(binary_stream)


@pytest.mark.parametrize("binary_stream", [_zip_builder], indirect=True)
def test_zip(
    binary_stream: ZipStream,
    lhs_stream: TableStream,
    rhs_stream: TableStream,
):
    for result in binary_stream:
        assert binary_stream.idx == lhs_stream.idx == rhs_stream.idx
        concat = tensordict.merge_tensordicts(
            lhs_stream[lhs_stream.idx - 1], rhs_stream[rhs_stream.idx - 1]
        )
        assert (result == concat).all()


def _join_builder(
    lhs_stream: TableStream, rhs_stream: TableStream
) -> NestedLoopJoinStream:
    return NestedLoopJoinStream(left=lhs_stream, right=rhs_stream, key="i1d")


@pytest.mark.parametrize("binary_stream", [_join_builder], indirect=True)
def test_join_input_len(
    binary_stream: NestedLoopJoinStream,
    lhs_stream: TableStream,
    rhs_stream: TableStream,
):
    assert len(binary_stream) == len(lhs_stream) * len(rhs_stream)


@pytest.mark.parametrize("binary_stream", [_join_builder], indirect=True)
def test_join_equal_as_original(
    binary_stream: NestedLoopJoinStream,
    lhs_stream: TableStream,
    rhs_stream: TableStream,
):
    block_frame_block = tensordict.cat(list(lhs_stream))
    joinable_frame_block = tensordict.cat(list(rhs_stream))

    # Performing the join here.
    results: list[TensorDict] = list(binary_stream)
    assert len(results), "The binary stream is empty."
    answer_items = tensordict.cat(results)["i1d"]

    # Do it at once, using ``TorchListTable`` as it yields everything in 1 batch.
    ground_truth = tensordict.cat(
        list(
            NestedLoopJoinStream(
                left=TableStream(TensorDictListTable([block_frame_block])),
                right=TableStream(TensorDictListTable([joinable_frame_block])),
                key="i1d",
            )
        )
    )

    answer_count = Counter(answer_items.tolist())
    truth_count = Counter(ground_truth["i1d"].tolist())

    assert answer_count == truth_count


@pytest.mark.parametrize("binary_stream", [_join_builder], indirect=True)
def test_match_functionally(
    binary_stream: NestedLoopJoinStream,
    lhs_stream: TableStream,
    rhs_stream: TableStream,
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
    assert len(binary_stream), "The binary stream is empty."

    assert binary_stream.idx == 0, "Pre iteration stream's index starts with 0."

    batches: list[TensorDict] = []
    for idx, batch in enumerate(binary_stream, start=1):
        # Ensure that the input is also exhausted.
        assert idx == binary_stream.idx
        batches.append(batch)

    assert binary_stream.idx == len(binary_stream) == len(batches)
