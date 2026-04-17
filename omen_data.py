from __future__ import annotations

import random
from typing import List, Sequence, Tuple

import torch


def _encode_template(template: str, seq_len: int) -> torch.Tensor:
    encoded = [byte % 256 for byte in template.encode("utf-8")]
    if len(encoded) < seq_len:
        encoded = encoded + [0] * (seq_len - len(encoded))
    else:
        encoded = encoded[:seq_len]
    return torch.tensor(encoded, dtype=torch.long)


def make_counting(n: int, seq_len: int) -> List[torch.Tensor]:
    data: List[torch.Tensor] = []
    for _ in range(n):
        start = random.randint(1, 50)
        delta = random.randint(1, 5)
        data.append(torch.tensor([(start + idx * delta) % 200 + 10 for idx in range(seq_len)]))
    return data


def make_python(n: int, seq_len: int) -> List[torch.Tensor]:
    templates = [
        "def add(a, b):\n    return a + b\n",
        "x = 0\nfor i in range(10):\n    x += i\n",
        "class Node:\n    def __init__(self, v):\n        self.v = v\n",
        "if x > 0:\n    print(x)\nelse:\n    print(-x)\n",
        "def fib(n):\n    if n<2: return n\n    return fib(n-1)+fib(n-2)\n",
    ]
    return [_encode_template(random.choice(templates), seq_len) for _ in range(n)]


def make_javascript(n: int, seq_len: int) -> List[torch.Tensor]:
    templates = [
        "function add(a, b) {\n  return a + b;\n}\n",
        "let total = 0;\nfor (const x of [1, 2, 3]) {\n  total += x;\n}\n",
        "class Node {\n  constructor(v) {\n    this.v = v;\n  }\n}\n",
        "const value = x > 0 ? x : -x;\nconsole.log(value);\n",
        "const fib = (n) => n < 2 ? n : fib(n - 1) + fib(n - 2);\n",
    ]
    return [_encode_template(random.choice(templates), seq_len) for _ in range(n)]


def make_rust(n: int, seq_len: int) -> List[torch.Tensor]:
    templates = [
        "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n",
        "fn total(xs: &[i32]) -> i32 {\n    xs.iter().sum()\n}\n",
        "struct Node {\n    v: i32,\n}\n",
        "fn abs_like(x: i32) -> i32 {\n    if x > 0 { x } else { -x }\n}\n",
        "fn fib(n: i32) -> i32 {\n    if n < 2 { n } else { fib(n - 1) + fib(n - 2) }\n}\n",
    ]
    return [_encode_template(random.choice(templates), seq_len) for _ in range(n)]


def make_multilingual_code(n: int, seq_len: int) -> List[torch.Tensor]:
    families = (make_python, make_javascript, make_rust)
    chunks: List[torch.Tensor] = []
    per_family = max(n // len(families), 1)
    for generator in families:
        chunks.extend(generator(per_family, seq_len))
    while len(chunks) < n:
        chunks.extend(random.choice(families)(1, seq_len))
    random.shuffle(chunks)
    return chunks[:n]


def make_rule_transfer(n: int, seq_len: int) -> List[torch.Tensor]:
    data: List[torch.Tensor] = []
    for _ in range(n):
        left = random.randint(10, 50)
        right = random.randint(10, 50)
        op = random.choice([0, 1, 2])
        if op == 0:
            result = (left + right) % 200 + 10
        elif op == 1:
            result = abs(left - right) + 10
        else:
            result = (left * right) % 200 + 10
        seq = [left, 100 + op, right, 200, result] + [0] * max(seq_len - 5, 0)
        data.append(torch.tensor(seq[:seq_len], dtype=torch.long))
    return data


def collate(batch: Sequence[torch.Tensor] | Sequence[Tuple[torch.Tensor, torch.Tensor]]):
    if isinstance(batch[0], (tuple, list)):
        src = torch.stack([item[0] for item in batch])
        tgt = torch.stack([item[1] for item in batch])
        return src, tgt
    stacked = torch.stack(batch)  # type: ignore[arg-type]
    return stacked[:, :-1], stacked[:, 1:]
