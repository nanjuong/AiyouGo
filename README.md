# aiyougo-compression

模型与数据的轻量级压缩技术（vendor 自 [AetherFlow](https://github.com/aiyougo/aetherflow)），解决两件事：**不物化稠密 f32 权重**、**不驻留全量语料**。

仅依赖 Python 标准库 + numpy。自带 IR 层与 CPU 参考后端，`demo.py` / `test.py` 开箱即用。

English documentation: [README.en.md](README.en.md)

## 安装

```bash
pip install .
# 或开发模式
pip install -e .
```

## 包结构

```
aetherflow/
├── __init__.py            # 顶层导出 Graph / backward / f32 / i32 / ReferenceBackend
├── ir/                    # ★ 中间表示层（本包约 70% 的代码在这，见下）
├── backends/              # 后端抽象 + CPU numpy 参考后端
├── runtime/               # 符号绑定执行器（Executor / ReferenceExecutor）
├── models/                # 字节级 Mamba 语言模型构图（压缩 pass 的演示载荷）
├── passes/
│   ├── base.py            # Pass / PassManager 图改写协议
│   └── param_compress.py  # ★ 参数压缩图改写 pass（low_rank / qlora）
└── train/
    ├── compress.py        # ★ 训练侧压缩 API（compress_for_training 等）
    ├── corpus.py          # ★ token 流无损压缩 + TokenCorpus 惰性解码
    ├── optim.py           # SGD / Adam（槽位跟着参数走，这是内存判据的前提）
    ├── params.py          # ParameterStore
    ├── grads.py           # clip_global_norm / global_norm
    └── mini_trainer.py    # 极简 Trainer（前向+反向+裁剪+一步更新）
```

★ 是压缩本体；其余是让它可运行、可验证的支撑层。

### IR 层（`aetherflow.ir`）

压缩 pass 作用在**计算图**上而不是张量上，因此本包自带完整 IR 层——这是理解代码结构的钥匙：

| 文件 | 作用 |
|---|---|
| `ir/graph.py` | 不可变计算图：SSA 值 + 节点 + revision 缓存失效 |
| `ir/types.py` | 符号维类型系统（`f32[4, -1, 512]` 这类部分已知形状） |
| `ir/ops.py` | 57 个算子：形状推断 + numpy 参考语义，注册式扩展 |
| `ir/grad.py` + `ir/grad_rules.py` | `backward(graph, wrt=…)`：42 条 vjp 规则挂算子，生成反向图 |
| `ir/second_order.py` | HVP（hessian-vector product） |
| `ir/control.py` | scan/cond 控制流构造器（因果卷积、选择性扫描由它展开） |
| `ir/liveness.py` | 静态显存预算（复用池 + 峰值驻留） |
| `ir/serialize.py` + `ir/version.py` | JSON 序列化与 IR 版本迁移（只升不降） |
| `ir/verifier.py` | 构图后静态检查（形状、dtype、可移植性等级） |

`backends/` 与 `runtime/` 提供 CPU 参考执行器：所有数值判据（误差界、roundtrip、
收敛性）都在它上面跑通，GPU 后端（OpenCL/CUDA）以它为真值对照。

## 核心功能

### 1. 参数压缩（低秩）

`(m, n)` f32 权重 `W` → `A(m,r) @ B(r,n)`。可训练量 `m·n` → `(m+n)·r`；
Adam 的 m/v 槽位跟着参数走，同比例缩小。

```python
import numpy as np
from aetherflow.train.compress import low_rank_factors

w = np.random.randn(1024, 1024).astype(np.float32)
A, B = low_rank_factors(w, rank=32)
# 可训练 f32：4,194,304 B -> 262,144 B（16x）
```

**适用条件（重要）**：低秩的收益取决于权重奇异值谱的能量集中度。谱平坦的
矩阵（纯随机高斯）近满秩，低秩近似误差大——demo 第一行会打印能量占比，
先量谱，再决定用不用。

### 2. qlora：int8 底 + 低秩适配器

冻结的 int8 底（1 字节/元素）+ 每列 f32 scale + 可训练低秩适配器。
适配器学的恰好是量化丢掉的那部分（对残差 `W - dequant(base)` 做 SVD），避免双重计数。

```python
from aetherflow.train.compress import quantize_int8

w = np.random.randn(1024, 1024).astype(np.float32)
base, scale = quantize_int8(w)   # base: int8 (m,n), scale: f32 (n,)
```

**真实驻留账**（1024×1024、r=32 实测）：

| 形态 | 驻留 | 相对稠密 f32 |
|---|---|---|
| 稠密 f32 | 4096 KB | 1× |
| qlora（int8 底 1024 KB + scale 4 KB + 适配器 256 KB） | 1284 KB | **3.2×** |
| low_rank（无底） | 256 KB | 16× |

注意两点：qlora 的驻留不止 int8 底，scale 与适配器都要算进来；且当
`m·n` 不太大时（约 87×87 以下）qlora 反而**膨胀**——底(1B)+适配器(8B/对)
压不过原矩阵(4B)。

**性能边界（按后端区分，别混为一谈）**：收益是**显存驻留与 Adam 槽位**，
不必然是墙钟时间。numpy 参考后端实测（1024×1024，r=32）：

| 形态 | 前向耗时 | 相对稠密 |
|---|---|---|
| 稠密 f32 单次 matmul | 186 ms | 1× |
| low_rank（1024×32 两次 matmul） | 0.5 ms | **快约 9×**（FLOPs 真降） |
| qlora（dequant 物化后再乘） | 435 ms | **慢约 1.7-5×** |

qlora 在参考后端变慢的原因很具体：numpy 必须先把 `cast + mul` 物化成一份
`m×n` 的 f32 才能乘，那正是压缩层想省掉的那份内存。int8 的带宽红利只有把
dequant **融进内核**才拿得到——那是内核层的工作，不是这一层的承诺。

### 3. token 流无损压缩（delta → zigzag → LEB128）

```python
from aetherflow.train.corpus import compress_tokens, decompress_tokens

tokens = [ord(c) for c in "hello world"]
blob = compress_tokens(tokens)          # 字节，无损
assert decompress_tokens(blob) == tokens
```

**收益目标要说清楚**：这条压缩对标的基线是训练中驻留的 **i32 token 数组
（4 B/元素）或 one-hot（V×4 B/元素）**，不是通用打包器：

| 输入 | 相对 i32 基线 | 说明 |
|---|---|---|
| 文本字节流（相邻相关） | ~3-9× | 字符级 LM 的真实形态 |
| 递增 id 序列 | ~30× | 长度排序语料的典型形态 |
| 随机字节 | **膨胀 1.6×** | delta 无事可做；demo 里有这条负面结论 |

### 4. TokenCorpus：只存压缩字节，按批惰性解码

```python
from aetherflow.train.corpus import TokenCorpus

# 示例语料：16 条序列，每条 128 个 token（token 取值须 < vocab_size）
sequences = [[i % 90 for i in range(128)] for _ in range(16)]
corpus = TokenCorpus(sequences, vocab_size=128)
for batch in corpus.iter_minibatches(4, window=96):
    # batch 含 ids/targets/mask/reset；只有当前批次的窗口被解压驻留
    ids = batch["ids"]        # (4, 96)
print(f"存储：{corpus.stored_bytes / 1e6:.4f} MB")
```

## 三条内存判据

对外的承诺以可验证的界给出，不是形容词（全部钉在 `test.py` 里）：

1. **可训练量**：`W (m,n) f32` → `A (m,r) + B (r,n)`，优化器槽位同比例缩；
   qlora 的 int8 底与 scale 不进参数表，零槽位。
2. **驻留表示即真相**：参数表里找不到任何 `m×n` f32 稠密副本
   （`assert_dense_free` 运行时检查，混进副本会报错）。
3. **数据侧**：语料以压缩字节驻留，`iter_minibatches` 按批惰性解码，
   全量 one-hot / 解压副本从不出现；无损性由 roundtrip 判据钉死。

误差界同样是可算的：低秩改写后的前向差 = 丢弃的奇异值（SVD 截断界），
量化误差 ≤ 半格点 × scale，均与手算一致（见 demo 输出）。

## 运行演示与测试

```bash
python demo.py              # 三幕：参数压缩（含误差界+能量占比）/ token 压缩（含负面结论）/ 端到端训练+生成
python -m pytest test.py    # 64 个测试：roundtrip、SVD界、容量-秩单调性、冻结量逐位不动、收敛性
```

## 依赖

- Python >= 3.8
- numpy >= 1.20

## 许可证

CC-BY-4.0。详见 [LICENSE](LICENSE)。

## 作者

aiyougo

## 相关链接

- [README.en.md](README.en.md) — 英文文档
- 主项目：AetherFlow（字节级 Mamba 语言模型，Intel Arc OpenCL 训练）
