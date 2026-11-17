import heapq
from dataclasses import dataclass, field
from typing import Optional


@dataclass(order=True)
class HuffmanNode:
    freq: int
    symbol: Optional[str] = field(compare=False, default=None)
    left: Optional["HuffmanNode"] = field(compare=False, default=None)
    right: Optional["HuffmanNode"] = field(compare=False, default=None)

class HuffmanTree:
    def __init__(self, frequencies: dict[str, int]) -> None:
        heap = [HuffmanNode(freq=f, symbol=s) for s, f in frequencies.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            heapq.heappush(heap, HuffmanNode(freq=lo.freq + hi.freq, left=lo, right=hi))
        root = heap[0]

        self.codes: dict[str, str] = {}
        self.paths: dict[str, tuple[tuple[int, int], ...]] = {}
        self.node_count = 0

        def build_paths(
            node: HuffmanNode,
            code: str = "",
            path: tuple[tuple[int, int], ...] = (),
        ) -> None:
            if node.left is None and node.right is None:
                if node.symbol is not None:
                    self.codes[node.symbol] = code or "0"
                    self.paths[node.symbol] = path
                return

            node_index = self.node_count
            self.node_count += 1
            if node.left is not None:
                build_paths(node.left, code + "0", path + ((node_index, 0),))
            if node.right is not None:
                build_paths(node.right, code + "1", path + ((node_index, 1),))

        build_paths(root)
