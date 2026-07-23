import bisect
import math
from typing import Iterable

from app.config import BLOOM_FILTER_CAPACITY, BLOOM_FILTER_FALSE_POSITIVE_RATE


def _bloom_num_bits(capacity: int, false_positive_rate: float) -> int:
    return max(64, math.ceil(-(capacity * math.log(false_positive_rate)) / (math.log(2) ** 2)))


def _bloom_num_hashes(num_bits: int, capacity: int) -> int:
    return max(1, round((num_bits / capacity) * math.log(2)))


class ContentHashIndex:
    """Dedup check for sha256 content hashes (hex strings), backed by a
    bloom filter for fast rejection of hashes we definitely haven't seen,
    falling back to binary search over a sorted list for a definitive
    answer whenever the bloom filter says "maybe". Never produces a false
    negative — at worst, a bloom-filter false positive costs one binary
    search rather than skipping real work."""

    def __init__(
        self,
        existing_hashes: Iterable[str] = (),
        capacity: int = BLOOM_FILTER_CAPACITY,
        false_positive_rate: float = BLOOM_FILTER_FALSE_POSITIVE_RATE,
    ) -> None:
        self._num_bits = _bloom_num_bits(capacity, false_positive_rate)
        self._num_hashes = _bloom_num_hashes(self._num_bits, capacity)
        self._bloom = bytearray((self._num_bits + 7) // 8)

        # Bulk-load via sort (O(n log n), Timsort) rather than repeated
        # bisect.insort (O(n) per insert -> O(n^2) overall) — matters once a
        # model has accumulated thousands of images across many crawl runs.
        hash_ints = [int(h, 16) for h in existing_hashes]
        for value in hash_ints:
            self._set_bloom_bits(value)
        self._sorted_hashes: list[int] = sorted(hash_ints)

    def add(self, content_hash: str) -> None:
        value = int(content_hash, 16)
        self._set_bloom_bits(value)
        bisect.insort(self._sorted_hashes, value)

    def contains(self, content_hash: str) -> bool:
        value = int(content_hash, 16)
        if not self._might_contain(value):
            return False
        idx = bisect.bisect_left(self._sorted_hashes, value)
        return idx < len(self._sorted_hashes) and self._sorted_hashes[idx] == value

    def _positions(self, value: int):
        # Double hashing (Kirsch-Mitzenmacher): derive k positions from two
        # 64-bit halves of the existing sha256 digest instead of computing k
        # independent hashes from scratch.
        h1 = value & 0xFFFFFFFFFFFFFFFF
        h2 = (value >> 64) & 0xFFFFFFFFFFFFFFFF
        if h2 == 0:
            h2 = 1
        num_bits = self._num_bits
        return ((h1 + i * h2) % num_bits for i in range(self._num_hashes))

    def _set_bloom_bits(self, value: int) -> None:
        for pos in self._positions(value):
            self._bloom[pos // 8] |= 1 << (pos % 8)

    def _might_contain(self, value: int) -> bool:
        return all(self._bloom[pos // 8] & (1 << (pos % 8)) for pos in self._positions(value))
