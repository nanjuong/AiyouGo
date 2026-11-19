"""压缩态训练演示：参数内存与训练数据同时压缩，且**不需要解压就能训练**。

跑法（项目根目录）::

    python demo.py

三幕，每一幕都给真实数字，不给结论式的形容词：

1. **参数压缩**。一个 Transformer 权重矩阵，两种驻留形态：
   - ``low_rank``：W(m,n) f32 换成 A(m,r) + B(r,n)，可训练量 (m+n)r
   - ``qlora``：int8 底（1 字节/元素，冻结）+ 每列 scale（冻结）+ 秩 r 适配器（可训练）
   展示 IR 里的实际节点形态（dequant 是图内的 ``cast`` + ``mul``，不是主机侧还原），
   以及参数表里**没有任何一份稠密 f32 副本**这件事。
2. **数据压缩**。token 流按 delta+zigzag+varint 无损存储；训练时按批惰性解码。
   对照"整份语料 one-hot 展开"的字节数——那正是被消掉的驻留对象。
3. **端到端**。一个 byte 级小 LM 在压缩形态下训练（Adam、反向图、优化器槽位
   全程只见因子），loss 收敛，最后把驻留形态重建成稠密权重交给推理引擎，
   让它续写 π。
"""

import time

import numpy as np

from aetherflow import Graph, ReferenceBackend, f32, i32
from aetherflow.models.lm import (
    LMSpec,
    build_language_model,
    generate,
    masked_lm_loss_indexed,
)
from aetherflow.passes.param_compress import ParamCompression, plan_param_compression
from aetherflow.train import (
    Adam,
    ParameterStore,
    TokenCorpus,
    Trainer,
    assert_dense_free,
    compress_for_training,
    low_rank_factors,
    memory_report,
    quantize_int8,
)

BACKEND = ReferenceBackend()
PI = (
    "3141592653589793238462643383279502884197169399375105820974944592"
    "3078164062862089986280348253421170679"
)
TOKENS = [ord(c) - 48 for c in PI]


def kb(n):
    return f"{n / 1024:8.1f} KB"


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------
rule("第一幕　参数压缩：稠密 f32 权重换成交换机形态")
# --------------------------------------------------------------------------
m, n, r = 1024, 1024, 32
g = Graph("linear")
x = g.input("x", f32[8, m])
w = g.input("w", f32[m, n])
g.output(g.matmul(x, w), "y")

# 权重用**几何衰减的奇异值谱**：0.96^k，前 32 个携带 ~93% 能量。真实训练
# 出来的权重层大致落在这个量级（有效秩远小于名义维度），这是低秩可行的
# 前提。刻意保留对照事实：谱平坦的矩阵（比如纯随机高斯）近满秩，低秩在
# 其上必然丢掉绝大部分能量——那种矩阵本来就不该用低秩压，第一行打印的
# 能量占比就是这个选择的依据。
rng0 = np.random.default_rng(0)
U0, _, V0 = np.linalg.svd(rng0.standard_normal((m, n)), full_matrices=False)
decay = 0.96 ** np.arange(m)
W = ((U0 * decay[None, :]) @ V0).astype(np.float32)
sv_all = np.linalg.svd(W, compute_uv=False)
energy_r = float(np.sum(sv_all[:r] ** 2) / np.sum(sv_all ** 2))
print(f"奇异值谱：前 {r}/{m} 个携带 {energy_r * 100:.1f}% 能量（决定低秩能不能用）")

print(f"原始：W 是 ({m}, {n}) f32 = {kb(W.nbytes)}，加 Adam 两份动量共 {kb(3 * W.nbytes)}")

for mode in ("low_rank", "qlora"):
    cm = compress_for_training(g, {"w": W}, ranks={"w": r},
                               keep_base={"w": mode == "qlora"}, data_inputs=["x"])
    rep = memory_report({"w": W}, cm)
    print(f"\n[{mode}]  驻留 {kb(rep['compressed_bytes'])} / 原始 {kb(rep['original_bytes'])}"
          f"  → {rep['ratio']:.1f}×")
    print(f"    可训练: " + ", ".join(f"{k} {v.shape} f32" for k, v in cm.trainable.items()))
    if cm.frozen:
        print(f"    冻结  : " + ", ".join(
            f"{k} {v.shape} {str(v.dtype).replace('dtype(', '').replace(')', '')}"
            for k, v in cm.frozen.items()))
    names = [v.name for v in cm.graph.inputs]
    print(f"    IR 输入: {names}")
    ops = [nd.op.name for nd in cm.graph.nodes]
    print(f"    IR 节点: {ops}")
    # 展示：dequant 与适配器都在图里，主机侧交出去的是 int8 与因子
    probe = np.random.default_rng(1).standard_normal((8, m)).astype(np.float32)
    dense_out = BACKEND.execute(g, {"x": probe, "w": W})["y"]
    cmp_out = BACKEND.execute(cm.graph, {"x": probe, **cm.frozen, **cm.trainable})["y"]
    gap = float(np.linalg.norm(dense_out - cmp_out))
    scale = float(np.linalg.norm(dense_out))
    if mode == "qlora":
        base, w_scale = quantize_int8(W)
        target = W - base.astype(np.float32) * w_scale[None, :]   # 适配器拟合的是残差
    else:
        target = W
    # 两个都成立的上界，取更紧的那个报出去：
    #   ‖x E‖_F ≤ ‖x‖₂ · ‖E‖_F   （谱范数 × Frobenius）
    #   ‖x E‖_F ≤ ‖x‖_F · ‖E‖₂   （Frobenius 次乘性）
    # E = target − 分解，由 Eckart–Young，截断 SVD 是最佳秩 r 逼近，
    # ‖E‖_F = sqrt(Σ_{i>r} σᵢ²)、‖E‖₂ = σ_{r+1}，都能从奇异值直接算。
    sv = np.linalg.svd(target, compute_uv=False)
    fro = float(np.sqrt(np.sum(sv[r:] ** 2)))
    spec = float(sv[r]) if len(sv) > r else 0.0
    bound = min(float(np.linalg.norm(probe, 2) * fro),
                float(np.linalg.norm(probe, "fro")) * spec)
    print(f"    前向相对误差 = {gap / scale:.4f}，‖Δy‖ = {gap:.4f} ≤ 界 {bound:.4f} "
          f"{'✓' if gap <= bound * 1.02 else '✗ 越界!'}（界松 {(bound / max(gap, 1e-9)):.1f}×）")
    # 同机对照：压缩形态的前向不比稠密慢到不可接受
    t0 = time.perf_counter()
    for _ in range(5):
        BACKEND.execute(g, {"x": probe, "w": W})["y"]
    t_dense = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(5):
        BACKEND.execute(cm.graph, {"x": probe, **cm.frozen, **cm.trainable})["y"]
    t_cmp = time.perf_counter() - t0
    print(f"    前向耗时（参考后端，5 次）：稠密 {t_dense*1000:.1f}ms vs 压缩 {t_cmp*1000:.1f}ms"
          f"  → {t_cmp/t_dense:.2f}×")
    assert "w" not in names, "稠密 W 必须从图输入里消失"

# 参数表里没有任何稠密副本
cm = compress_for_training(g, {"w": W}, ranks={"w": r}, keep_base={"w": True},
                          data_inputs=["x"])
store = ParameterStore(cm.trainable)
assert_dense_free(store, cm.frozen, {"w": (m, n)})
print(f"\nassert_dense_free 通过：参数表 {sorted(store.names)} 里没有 'w'")
try:
    assert_dense_free(ParameterStore({**cm.trainable, "w": W}), cm.frozen, {"w": (m, n)})
except AssertionError as exc:
    print(f"混进稠密副本时它会响：{exc}")


# --------------------------------------------------------------------------
rule("第二幕　训练数据：无损字节存储 + 按批惰性解码")
# --------------------------------------------------------------------------
# 用真实文本当语料：one-hot 的放大倍数才有意义（V=258）
# 用内置说明文本当语料（demo 自包含，不依赖包外文件）：
# 自然语言文本相邻字节强相关，one-hot 的放大倍数才有意义（V=258）
text = (
    "低秩压缩把 m×n 的权重矩阵拆成 m×r 与 r×n 两个窄因子的乘积，"
    "可训练量从 m·n 降到 (m+n)·r；Adam 的一阶与二阶矩跟着参数走，"
    "槽位同比例缩小。qlora 在此基础上再垫一层冻结的 int8 底：底加每列"
    "scale 常驻但零槽位，可训练的只有低秩适配器，它学的恰好是量化丢掉的那部分。"
    "数据侧同理：token 流做 delta 编码后相邻差值变小，zigzag 把负数折成正数，"
    "LEB128 让小数只占一两个字节——无损，解压后与原始逐字节相等。"
) * 120
raw = text.encode("utf-8")
rows = [list(raw[i:i + 512]) for i in range(0, len(raw) - 512, 512)]
corpus = TokenCorpus(rows, 258)
rep = corpus.storage_report()
print(f"{corpus}")
print(f"  语料行数 {rep['sequences']}, token 数 {rep['tokens']:,}")
print(f"  压缩存储      {kb(rep['compressed_bytes'])}")
print(f"  i32 原样      {kb(rep['raw_i32_bytes'])}   （{rep['ratio_vs_raw']:.1f}×）")
print(f"  one-hot 全量  {kb(rep['onehot_all_bytes'])}   （{rep['ratio_vs_onehot']:.0f}×）")
print("  注：one-hot 才是训练里真正驻留的那份东西；压缩层消掉的是它。")

# 一条负面结论，避免被当成通用打包器
random_bytes = [int(b) for b in np.random.default_rng(0).integers(0, 256, 2000)]
from aetherflow.train import compress_tokens, decompress_tokens

blob = compress_tokens(random_bytes)
print(f"\n负面结论：随机字节 2000 个 → 压缩 {len(blob)} B（原始 2000 B）")
print(f"           delta 编码在这里**膨胀**了 {len(blob)/2000:.1f}×，但相对 i32 仍省 {4*2000/len(blob):.1f}×")
print(f"           无损性不变：{decompress_tokens(blob) == random_bytes}")
print("           → 这条压缩的收益目标是 one-hot / i32 表示，不是通用打包。")

t0 = time.perf_counter()
seen = 0
for batch in corpus.iter_minibatches(16, window=256, labels="index"):
    seen += batch["ids"].shape[0]
print(f"\n惰性遍历全量 {seen} 行、每批只解窗口，耗时 {time.perf_counter()-t0:.2f}s；"
      f"任意时刻驻留的只有 16×256 的窗口。")


# --------------------------------------------------------------------------
rule("第三幕　端到端：压缩态训练一个会续写 π 的小 LM")
# --------------------------------------------------------------------------
spec = LMSpec(vocab_size=10, d_model=24, num_layers=1, d_state=8, d_conv=4,
              steps=24, batch=4, tie_weights=True)
gl = Graph("lm")
_, logits = build_language_model(gl, spec, with_reset=True)
t_in = gl.input("targets", i32[4, 24])
m_in = gl.input("mask", f32[4, 24])
gl.output(masked_lm_loss_indexed(gl, logits, t_in, m_in), "loss")

DATA = {"ids", "targets", "mask", "reset"}
rng = np.random.default_rng(5)
weights = {
    v.name: (rng.standard_normal(tuple(int(d) for d in v.type.shape)) * 0.3).astype(np.float32)
    for v in gl.inputs
    if v.name not in DATA and v.type.dtype.value == "f32"
}
matmul_w = {
    a.name for node in gl.nodes if node.op.name == "matmul"
    for a in node.inputs if a.producer is None and len(a.type.shape) == 2
}
ranks = {k: min(4, min(weights[k].shape)) for k in weights
         if k in matmul_w and min(weights[k].shape) >= 2}
cm = compress_for_training(gl, weights, ranks=ranks,
                           keep_base={k: True for k in ranks}, data_inputs=DATA)
rep = memory_report(weights, cm)
print(f"被压缩的参数: {sorted(ranks)}（秩 {sorted(set(ranks.values()))}）")
print(f"参数驻留 {kb(rep['compressed_bytes'])} vs 稠密 {kb(rep['original_bytes'])}"
      f" → {rep['ratio']:.1f}×；其中 int8 底 {kb(rep['frozen_bytes'])} 是冻结的")

store = ParameterStore.from_graph(cm.graph, exclude=cm.exclude, seed=1)
for k, v in cm.trainable.items():
    store[k][...] = v
trainer = Trainer(cm.graph, store, Adam(store.arrays(), lr=0.05), output="loss")

seqs = [TOKENS[i:i + 25] for i in range(0, len(TOKENS) - 24, 25)]
seqs = seqs[: (len(seqs) // 4) * 4]
corp = TokenCorpus(seqs, 10)
frozen_before = {k: v.copy() for k, v in cm.frozen.items()}

t0 = time.perf_counter()
losses = []
for epoch in range(40):
    for batch in corp.iter_minibatches(4, window=24, labels="index"):
        losses.append(float(trainer.step(cm.feed(batch))))
    if epoch in (0, 5, 15, 39):
        print(f"  epoch {epoch:2d}: loss = {losses[-1]:.4f}")
print(f"训练 {len(losses)} 个微批，用时 {time.perf_counter()-t0:.1f}s")

moved = [k for k, v in cm.frozen.items() if not np.array_equal(v, frozen_before[k])]
print(f"冻结量被改写过吗：{moved or '没有——int8 底与 scale 全程不动'}")
print(f"优化器持有的可训练量：{sorted(store.names)}")

dense = cm.dense_weights(store.arrays())
gen = list(generate(BACKEND, dense, [3, 1, 4, 1, 5], spec.with_batch(1),
                    max_new_tokens=10, eos_id=255))
truth = TOKENS[5:15]
hits = sum(int(a == b) for a, b in zip(gen, truth))
print("\n推理侧（重建稠密后交给 generate）：")
print(f"  prompt   : {PI[:5]}")
print(f"  模型续写 : {''.join(str(t) for t in gen)}")
print(f"  π 的真值 : {PI[5:15]}")
print(f"  命中 {hits}/10")

rule("小结")
print("""
1. 参数：稠密 f32 权重从图输入与参数表里**消失**，驻留的是因子（可训练）
   与 int8 底（冻结）；dequant 是 IR 里的 cast+mul，GPU 后端可融合进内核。
2. 数据：语料以无损字节驻留，按批惰性解码；被消掉的是 one-hot 全量物化。
3. 训练与推理对接：dense_weights(store.arrays()) 一步重建，走原推理引擎。
4. 判据不是"能跑"，而是可验证的界：SVD 截断界、量化半格点界、
   梯度对链式闭式解 1e-3、冻结量逐位不动、生成对真值的命中率。

边界（同样重要）：
  · **速度取决于后端**。本演示在 numpy 参考后端实测：low_rank 的前向反而
    快约 9×（1024×1024 的单次 matmul 拆成 1024×32 两次，FLOPs 真降）；
    qlora 慢约 5×（numpy 要先把 1024×1024 的 dequant 物化出来再乘，
    cast+mul 都是实打实的内存操作）。int8 驻留的带宽红利只有把 dequant
    融进内核才拿得到——那是要在内核层做的事，不是这层的承诺。
  · 秩/是否留 int8 底要按层选，本演示用统一小秩；真实模型该由调优层决定。
  · 低秩能不能用，看奇异值谱：能量占比打印在第一行。谱平坦的近满秩矩阵
    （纯随机高斯就是）不该用低秩压。
  · qlora 的可训练容量只在适配器上，rank 太小会欠拟合——见 test.py 里 lowrank 的容量-秩单调性判据。
  · delta 编码对随机字节是膨胀的，它服务的是 one-hot / i32 场景。
  · 静态形状框架：每个 (B,T) 组合一张图，长上下文仍受构图成本限制。
""")
