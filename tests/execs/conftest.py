# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls


@dcls.dataclass
class FakeIter:
    source: "FakeRange"
    idx: int = 0

    def __next__(self):
        try:
            return self._get_next()
        except IndexError:
            raise StopIteration

    def _get_next(self):
        result = self.source[self.idx]
        self.idx += 1
        return result


@dcls.dataclass
class FakeRange:
    length: int

    def __iter__(self) -> FakeIter:
        return FakeIter(self)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> int:
        l = self.length
        if not -l <= idx < l:
            raise IndexError

        return idx % l
