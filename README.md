# aiyougo-compression

Lightweight compression for models and data (vendored from
[AetherFlow](https://github.com/aiyougo/aetherflow)), solving two problems:
**never materialize dense f32 weights**, and **never keep the full corpus resident**.

Depends only on the Python standard library + numpy. Ships its own IR layer and a
CPU reference backend, so `demo.py` / `test.py` run out of the box.

中文文档见 [README.zh-cn.md](README.zh-cn.md)。

## Installation

```bash
pip install .
# or development mode
pip install -e .
```

## Package layout

```
aetherflow/
├── __init__.py            # top-level exports: Graph / backward / f32 / i32 / ReferenceBackend
├── ir/                    # ★ the IR layer (~70% of this package's code, see below)
├── backends/              # backend protocol + CPU numpy reference backend
├── runtime/               # symbol-binding executor (Executor / ReferenceExecutor)
├── models/                # byte-level Mamba LM graph builder (the demo payload for the pass)
├── passes/
│   ├── base.py            # Pass / PassManager graph-rewrite protocol
│   └── param_compress.py  # ★ parameter-compression graph pass (low_rank / qlora)
└── train/
    ├── compress.py        # ★ training-side compression API (compress_for_training, …)
    ├── corpus.py          # ★ lossless token-stream compression + lazily decoded TokenCorpus
    ├── optim.py           # SGD / Adam (slots follow the parameters — precondition of the bounds)
    ├── params.py          # ParameterStore
    ├── grads.py           # clip_global_norm / global_norm
    └── mini_trainer.py    # minimal Trainer (forward + backward + clip + step)
```

★ marks the compression itself; the rest is the support layer that makes it runnable and verifiable.

### The IR layer (`aetherflow.ir`)

The compression pass rewrites a **compute graph**, not tensors — which is why this
package ships a complete IR layer. This is the key to reading the code:

| File | Purpose |
|---|---|
| `ir/graph.py` | Immutable graph: SSA values + nodes + revision-based cache invalidation |
| `ir/types.py` | Symbolic-dimension type system (partially known shapes like `f32[4, -1, 512]`) |
| `ir/ops.py` | 57 operators: shape inference + numpy reference semantics, registry-style extension |
| `ir/grad.py` + `ir/grad_rules.py` | `backward(graph, wrt=…)`: 42 vjp rules attached to ops, emitting a backward graph |
| `ir/second_order.py` | HVP (hessian-vector product) |
| `ir/control.py` | scan/cond control-flow builders (causal conv and selective scan are expanded through it) |
| `ir/liveness.py` | Static memory budget (reuse pool + peak residency) |
| `ir/serialize.py` + `ir/version.py` | JSON serialization and IR version migration (upgrade only) |
| `ir/verifier.py` | Post-build static checks (shapes, dtypes, portability level) |

`backends/` and `runtime/` provide the CPU reference executor: every numerical
claim below (error bounds, roundtrips, convergence) is verified on it. GPU
backends (OpenCL/CUDA) treat it as ground truth.

## Features

### 1. Parameter compression (low rank)

An `(m, n)` f32 weight `W` becomes `A(m,r) @ B(r,n)`. Trainable volume goes from
`m·n` to `(m+n)·r`; Adam's m/v slots follow the parameters and shrink identically.

```python
import numpy as np
from aetherflow.train.compress import low_rank_factors

w = np.random.randn(1024, 1024).astype(np.float32)
A, B = low_rank_factors(w, rank=32)
# trainable f32: 4,194,304 B -> 262,144 B (16x)
```

**When it applies (important)**: low rank pays off only in proportion to how much
energy sits in the leading singular values. A flat-spectrum matrix (pure random
Gaussian) is nearly full rank and compresses badly — the demo prints the energy
fraction on its first line. Measure the spectrum before you commit to a rank.

### 2. qlora: int8 base + low-rank adapter

A frozen int8 base (1 byte/element) + per-column f32 scale + a trainable low-rank
adapter. The adapter is fit to the residual `W - dequant(base)`, i.e. it learns
exactly what quantization discarded, so nothing is double-counted.

```python
from aetherflow.train.compress import quantize_int8

w = np.random.randn(1024, 1024).astype(np.float32)
base, scale = quantize_int8(w)   # base: int8 (m,n), scale: f32 (n,)
```

**Honest residency accounting** (measured at 1024×1024, r=32):

| Form | Resident | vs dense f32 |
|---|---|---|
| dense f32 | 4096 KB | 1× |
| qlora (int8 base 1024 KB + scale 4 KB + adapter 256 KB) | 1284 KB | **3.2×** |
| low_rank (no base) | 256 KB | 16× |

Two caveats stated plainly: qlora's footprint is *not* just the int8 base — the
scale and the adapter count too; and below roughly 87×87 elements qlora
**inflates** rather than shrinks (1 B base + 8 B/element adapter cannot beat 4 B
dense).

**Performance boundary (backend-dependent — do not conflate)**: the win is
**VRAM residency and optimizer slots**, not necessarily wall-clock time.
Measured on the numpy reference backend (1024×1024, r=32):

| Form | Forward time | vs dense |
|---|---|---|
| dense f32, single matmul | 186 ms | 1× |
| low_rank (two 1024×32 matmuls) | 0.5 ms | **~9× faster** (real FLOP drop) |
| qlora (dequant materialized, then multiply) | 435 ms | **~1.7-5× slower** |

The reason qlora gets slower on the reference backend is concrete: numpy must
materialize `cast + mul` into an `m×n` f32 array before multiplying — exactly
the allocation the compression layer exists to avoid. The int8 bandwidth payoff
only lands once dequant is **fused into the kernel**; that is kernel-layer work,
not a promise of this layer.

### 3. Lossless token-stream compression (delta → zigzag → LEB128)

```python
from aetherflow.train.corpus import compress_tokens, decompress_tokens

tokens = [ord(c) for c in "hello world"]
blob = compress_tokens(tokens)          # bytes, lossless
assert decompress_tokens(blob) == tokens
```

**What the ratio is against**: the baseline is what training actually keeps
resident — an **i32 token array (4 B/element) or one-hot (V×4 B/element)** —
not general-purpose packaging:

| Input | vs i32 baseline | Note |
|---|---|---|
| text byte stream (locally correlated) | ~3-9× | the real shape for a character-level LM |
| increasing id sequences | ~30× | typical for length-sorted corpora |
| random bytes | **inflates 1.6×** | nothing for delta to exploit; the demo shows this negative result |

### 4. TokenCorpus: store compressed bytes only, decode per batch

```python
from aetherflow.train.corpus import TokenCorpus

# Sample corpus: 16 sequences of 128 tokens each (each token must be < vocab_size)
sequences = [[i % 90 for i in range(128)] for _ in range(16)]
corpus = TokenCorpus(sequences, vocab_size=128)
for batch in corpus.iter_minibatches(4, window=96):
    # batch holds ids/targets/mask/reset; only this batch's window is decompressed
    ids = batch["ids"]        # (4, 96)
print(f"stored: {corpus.stored_bytes / 1e6:.4f} MB")
```

## Three memory claims, each a checkable bound

Not adjectives — every one is pinned by a test in `test.py`:

1. **Trainable volume**: `W (m,n) f32` → `A (m,r) + B (r,n)`; optimizer slots
   shrink identically. In qlora the int8 base and scale never enter the
   parameter table, so they cost zero slots.
2. **Resident form is truth**: no dense `m×n` f32 copy exists anywhere in the
   parameter table (`assert_dense_free` checks this at runtime and fails loudly
   if a dense copy sneaks in).
3. **Data side**: the corpus resides as compressed bytes and `iter_minibatches`
   decodes per batch — the full one-hot (or a full decompressed copy) never
   appears; losslessness is pinned by roundtrip tests.

The error bounds are computable the same way: the forward difference after a
low-rank rewrite equals the discarded singular values (SVD truncation bound),
and quantization error ≤ half a lattice step × scale — both match hand
calculation (see the demo output).

## Run the demo and the tests

```bash
python demo.py              # three acts: parameter compression (bounds + energy fraction) / token compression (incl. negative result) / end-to-end training + generation
python -m pytest test.py    # 64 tests: roundtrip, SVD bound, capacity-vs-rank monotonicity, frozen bases bit-identical, convergence
```

## Dependencies

- Python >= 3.8
- numpy >= 1.20

## License

CC-BY-4.0. See [LICENSE](LICENSE).

## Author

aiyougo

## Related

- [README.zh-cn.md](README.zh-cn.md) — 中文文档
- Upstream: AetherFlow (byte-level Mamba language model, trained on Intel Arc via OpenCL)
