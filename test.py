"""压缩态训练的验收测试：参数压缩 + 数据压缩，端到端能学。

判据分四组，与 ``aetherflow/passes/param_compress.py``、
``aetherflow/train/compress.py``、``aetherflow/train/corpus.py`` 的承诺一一对应：

1. **编解码无损**（含边界与损坏检测）。同时钉住一条**负面结论**：delta
   编码对字节流相对原始字节是**膨胀**的——压缩收益的目标是 one-hot 物化
   与 i32 表示，不是"任何情况下都比原始字节小"。写在这里而不是藏在
   docstring 里，是为了让后来人不能顺手把它当成通用打包器用。
2. **低秩 / qlora 的数值等价性**：前向差有 SVD 截断界 / 量化格点界可验；
   因子梯度与链式闭式解一致（**不用有限差分当参照**：f32 下噪声底
   ≈0.12，会盖过真错误，闭式解才配得上 1e-3 的容差）。
3. **冻结与内存**：int8 底与 scale 不进优化器；参数表里没有被压参数的
   原名；槽位字节数随可训练量下降。
4. **端到端**：压缩态的小 LM 在 π 上确实收敛。

梯度判据选闭式而不是数值：``y = x @ A @ B`` 的 ``dA = xᵀ·dy·Bᵀ``、
``dB = Aᵀ·xᵀ·dy`` 是确定的矩阵乘法，与 VJP 的差只剩求和顺序；有限差分
在 f32 上的误差与真正的实现错误同量级，用它等于放弃判据。
"""

import unittest

import numpy as np

from aetherflow import Graph, ReferenceBackend, backward, f32, i32
from aetherflow.models.lm import LMSpec, build_language_model, masked_lm_loss_indexed
from aetherflow.passes.param_compress import (
    ParamCompression,
    compressed_inputs,
    plan_param_compression,
)
from aetherflow.train import Adam, ParameterStore, Trainer
from aetherflow.train.compress import (
    assert_dense_free,
    compress_for_training,
    low_rank_factors,
    memory_report,
    quantize_int8,
)
from aetherflow.train.corpus import (
    TokenCorpus,
    compress_tokens,
    decompress_tokens,
    peak_bytes,
)

BACKEND = ReferenceBackend()

# π 的前 120 位：数字流，词表 10，delta 小 → 压缩收益明显
PI = (
    "3141592653589793238462643383279502884197169399375105820974944592"
    "3078164062862089986280348253421170679"
)
PI_TOKENS = [ord(c) - 48 for c in PI]


def _mlp(m=48, n=32, batch=8):
    """单层线性：唯一参数 w 被 matmul 直接消费，是压缩的最小合法现场。"""
    g = Graph("mlp1")
    x = g.input("x", f32[batch, m])
    w = g.input("w", f32[m, n])
    g.output(g.matmul(x, w), "y")
    return g


def _weights(shape, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) / np.sqrt(shape[0])).astype(np.float32)


class TestTokenCodec(unittest.TestCase):
    def test_roundtrip_on_digit_stream(self):
        self.assertEqual(decompress_tokens(compress_tokens(PI_TOKENS)), PI_TOKENS)

    def test_roundtrip_on_edges(self):
        for seq in ([0], [7, 7, 7], [1000, 0, 1000], [5, 4, 3, 2, 1, 0],
                    [0, 10**9, 0], list(range(256))):
            with self.subTest(seq=str(seq)[:18]):
                self.assertEqual(decompress_tokens(compress_tokens(seq)),
                                 [int(x) for x in seq])

    def test_empty_stream_is_encodable(self):
        blob = compress_tokens([])
        self.assertEqual(decompress_tokens(blob), [])

    def test_truncated_stream_is_detected_not_padded(self):
        blob = compress_tokens([1000, 2000, 3000])
        with self.assertRaises(ValueError):
            decompress_tokens(blob[:-1])

    def test_trailing_garbage_is_detected(self):
        blob = compress_tokens([1, 2, 3]) + b"\x07"
        with self.assertRaises(ValueError):
            decompress_tokens(blob)

    def test_negative_delta_roundtrips_across_width_boundary(self):
        """zigzag 的奇偶与 varint 的进位边界最容易错，专门打一针。"""
        seq = [0, -1, 1, 63, 64, 65, 127, 128, 2**16, -(2**16), 7]
        self.assertEqual(decompress_tokens(compress_tokens(seq)), seq)

    def test_small_delta_stream_beats_i32(self):
        blob = compress_tokens(PI_TOKENS)
        self.assertLess(len(blob), 4 * len(PI_TOKENS))

    def test_dense_byte_stream_expands_relative_to_raw_bytes(self):
        """**负面结论，有意保留**：delta 编码对字节流是膨胀的（约 2×）。

        相邻字节之差通常 > 63，zigzag varint 要两个字节；而字节数组本身
        每 token 就一个字节。所以"压缩字节 vs 原始字节"这条比值上它**赢不了**，
        只有在跟 i32 数组（4B/token）或 one-hot（V×4B/token）比的时候才有收益。

        把这条写进测试而不是 docstring，是因为一个"压缩层"如果被误当通用
        打包器用，就会出现"压缩后变大还不知道"的事故——判据必须让这种情况
        在 CI 里响。
        """
        rng = np.random.default_rng(0)
        seq = [int(b) for b in rng.integers(0, 256, size=2000)]
        blob = compress_tokens(seq)
        self.assertGreater(len(blob), len(seq))          # 相对原始字节：膨胀
        self.assertLess(len(blob), 4 * len(seq))         # 相对 i32：仍然赚
        self.assertEqual(decompress_tokens(blob), seq)   # 但无损不变


class TestTokenCorpus(unittest.TestCase):
    def setUp(self):
        self.seqs = [PI_TOKENS[i:i + 40] for i in range(0, 100, 40)]
        self.corpus = TokenCorpus(self.seqs, 10)

    def test_construction_drops_the_uncompressed_lists(self):
        corpus = TokenCorpus(self.seqs, 10)
        self.assertFalse(hasattr(corpus, "_sequences"))
        self.assertEqual(set(vars(corpus)) & {"_blobs", "_lengths"},
                         {"_blobs", "_lengths"})
        for i, s in enumerate(self.seqs):
            self.assertEqual(corpus.decode(i), s)

    def test_rejects_out_of_range_tokens_and_short_rows(self):
        with self.assertRaises(ValueError):
            TokenCorpus([[0, 1, 25]], 10)
        with self.assertRaises(ValueError):
            TokenCorpus([[7]], 10)
        with self.assertRaises(ValueError):
            TokenCorpus([], 10)

    def test_batch_targets_are_the_right_shift(self):
        batch = next(iter(self.corpus.iter_minibatches(2, labels="index")))
        row = self.corpus.decode(0)
        t = batch["ids"].shape[1]
        np.testing.assert_array_equal(batch["ids"][0], np.array(row[:t]))
        V = self.corpus.vocab_size
        # index 口径的 targets 是扁平下标 (b*T + t)*V + y
        b = batch["ids"].shape[0]
        offsets = np.arange(b)[:, None] * t + np.arange(t)[None, :]
        np.testing.assert_array_equal(batch["targets"] // V, offsets)
        np.testing.assert_array_equal(batch["targets"] % V,
                                      np.array([row[1:1 + t],
                                                self.corpus.decode(1)[1:1 + t]]))

    def test_onehot_labels_are_onehot(self):
        batch = next(iter(self.corpus.iter_minibatches(2, labels="onehot")))
        oh = batch["targets"]
        self.assertTrue(np.allclose(oh.sum(-1), 1.0))
        row = self.corpus.decode(0)
        np.testing.assert_array_equal(oh.argmax(-1)[0], np.array(row[1:1 + oh.shape[1]]))

    def test_reset_marks_sequence_start_only(self):
        batch = next(iter(self.corpus.iter_minibatches(2, labels="index")))
        reset = batch["reset"]
        self.assertTrue(np.all(reset[:, 0] == 1.0))
        self.assertTrue(np.all(reset[:, 1:] == 0.0))

    def test_mask_covers_every_supervised_position(self):
        batch = next(iter(self.corpus.iter_minibatches(2, labels="index")))
        self.assertTrue(np.all(batch["mask"] == 1.0))

    def test_storage_report_compares_against_onehot(self):
        rep = self.corpus.storage_report()
        # one-hot 全量 = tokens × V × 4B，必然远大于压缩字节
        self.assertEqual(rep["onehot_all_bytes"], rep["tokens"] * 10 * 4)
        self.assertGreater(rep["ratio_vs_onehot"], 10)

    def test_window_api_rejects_overrun(self):
        with self.assertRaises(ValueError):
            self.corpus.window(0, 30, 20)      # 需要第 51 位，只有 40 位
        with self.assertRaises(ValueError):
            self.corpus.window(0, 0, 0)

    def test_default_drops_the_incomplete_tail_batch(self):
        """静态形状框架的必然：默认丢掉不足 batch 的尾批。

        图的 batch 维在构图期写死，尾批行数对不上只会在执行器入口报
        ``ShapeMismatchError``——那离"行数不整除"太远，所以语料层默认兜住。
        """
        corpus = TokenCorpus(self.seqs, 10)      # 3 行
        sizes = [b["ids"].shape[0] for b in corpus.iter_minibatches(2, labels="index")]
        self.assertEqual(sizes, [2])             # 第 3 行作为尾批被丢弃

    def test_keep_tail_batch_yields_actual_rows_not_padded(self):
        """``drop_last=False`` 保留尾批时，行数必须是实际剩余，不能补零行。

        补零会把不存在的样本喂进模型（且零行还是合法 token 0）。这条
        判据与"丢不丢"无关，管的是"既然保留了，就别造假数据"。
        """
        corpus = TokenCorpus(self.seqs, 10)      # 3 行
        batches = list(corpus.iter_minibatches(2, labels="index", drop_last=False))
        self.assertEqual([b["ids"].shape[0] for b in batches], [2, 1])
        # 尾批那一行确实是第 3 条序列，不是零
        third = np.array(self.seqs[2][: corpus.steps])
        np.testing.assert_array_equal(batches[1]["ids"][0], third)
        self.assertFalse(np.all(batches[1]["ids"][0] == 0))

    def test_fewer_rows_than_batch_raises_instead_of_yielding_nothing(self):
        """行数不足一个 batch：默认必须**报错**，不是静默产出空迭代。

        空迭代比形状报错更糟——它看起来"跑完了"，实际一个微批都没训。
        ``drop_last`` 允许丢弃**尾批**，但不允许把**全部**丢掉还不响。
        """
        corpus = TokenCorpus([self.seqs[0]], 10)     # 1 行，batch=4
        with self.assertRaisesRegex(ValueError, "不足 batch"):
            list(corpus.iter_minibatches(4, labels="index"))
        # 显式保留时不报错：由调用方自己决定那张残批图怎么构
        kept = list(corpus.iter_minibatches(4, labels="index", drop_last=False))
        self.assertEqual([b["ids"].shape[0] for b in kept], [1])


class TestLowRankRewrite(unittest.TestCase):
    def setUp(self):
        self.g = _mlp()
        self.W = _weights((48, 32), seed=1)
        self.X = np.random.default_rng(2).standard_normal((8, 48)).astype(np.float32)
        self.DY = np.random.default_rng(3).standard_normal((8, 32)).astype(np.float32)

    def test_policy_defaults_to_low_rank(self):
        plan = plan_param_compression(self.g, ranks={"w": 6})
        self.assertEqual(plan["w"], ("low_rank", 6))

    def test_dense_param_input_is_replaced_by_factors(self):
        gc = ParamCompression({"w": ("low_rank", 6)}).run(self.g)
        names = [v.name for v in gc.inputs]
        self.assertEqual(names, ["x", "w__A", "w__B"])
        self.assertEqual(compressed_inputs(gc), {"w": "low_rank"})

    def test_forward_gap_is_bounded_by_svd_truncation(self):
        r = 6
        A, B = low_rank_factors(self.W, r)
        gc = ParamCompression({"w": ("low_rank", r)}).run(self.g)
        ref = BACKEND.execute(self.g, {"x": self.X, "w": self.W})["y"]
        got = BACKEND.execute(gc, {"x": self.X, "w__A": A, "w__B": B})["y"]
        s = np.linalg.svd(self.W, compute_uv=False)
        bound = float(np.sqrt(np.sum(s[r:] ** 2)))
        # |x(A Bᵀ - W)| ≤ ‖x‖₂ · ‖ABᵀ-W‖_F，‖x‖₂ 是最大奇异值
        gap = float(np.max(np.abs(ref - got)))
        self.assertLessEqual(gap, np.linalg.norm(self.X, 2) * bound * 1.01)

    def test_full_rank_compression_is_exact(self):
        """秩取 min(m,n) 时低秩替换必须无损——这是分解本身正确性的判据。"""
        A, B = low_rank_factors(self.W, min(48, 32))
        np.testing.assert_allclose(A @ B, self.W, rtol=1e-4, atol=1e-5)

    def test_factors_only_are_trainable(self):
        gc = ParamCompression({"w": ("low_rank", 6)}).run(self.g)
        bwd = backward(gc, wrt=[gc.find("w__A"), gc.find("w__B")])
        self.assertEqual(sorted(bwd.output_names), ["dw__A", "dw__B"])
        A, B = low_rank_factors(self.W, 6)
        res = BACKEND.execute(bwd, {"x": self.X, "w__A": A, "w__B": B, "dy": self.DY})
        np.testing.assert_allclose(res["dw__A"], self.X.T @ self.DY @ B.T,
                                   rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(res["dw__B"], A.T @ self.X.T @ self.DY,
                                   rtol=1e-3, atol=1e-3)

    def test_rejects_param_consumed_outside_matmul(self):
        """bias 这类被 add 消费的一维参数不能被低秩替换——语义会变。"""
        g = Graph("bad")
        x = g.input("x", f32[8, 4])
        w = g.input("w", f32[4, 4])
        b = g.input("b", f32[4])
        g.output(g.add(g.matmul(x, w), b), "y")
        with self.assertRaisesRegex(ValueError, "不保语义"):
            plan_param_compression(g, ranks={"b": 2})
        # 同一条判据在 pass 层也必须响（调用方可能绕过 plan 直接构造）
        with self.assertRaisesRegex(ValueError, "只改写"):
            ParamCompression({"b": ("low_rank", 2)}).run(g)

    def test_rejects_rank_above_short_edge(self):
        with self.assertRaisesRegex(ValueError, "短边"):
            ParamCompression({"w": ("low_rank", 99)}).run(self.g)

    def test_unknown_param_name_is_refused_early(self):
        with self.assertRaises(KeyError):
            plan_param_compression(self.g, ranks={"wq": 2})

    def test_mode_typo_is_refused(self):
        with self.assertRaisesRegex(ValueError, "模式"):
            ParamCompression({"w": ("lowrank", 2)})

    def test_uncompressed_params_are_untouched(self):
        """没列进 policy 的参数必须原样存在——不能顺手压掉。"""
        g = Graph("two")
        x = g.input("x", f32[8, 4])
        w1 = g.input("w1", f32[4, 4])
        w2 = g.input("w2", f32[4, 4])
        g.output(g.add(g.matmul(x, w1), g.matmul(x, w2)), "y")
        gc = ParamCompression({"w1": ("low_rank", 2)}).run(g)
        names = [v.name for v in gc.inputs]
        self.assertIn("w2", names)
        self.assertNotIn("w1", names)
        w = _weights((4, 4), seed=4)
        ref = BACKEND.execute(g, {"x": self.X[:, :4], "w1": w, "w2": _weights((4, 4), seed=5)})["y"]
        A, B = low_rank_factors(w, 2)
        got = BACKEND.execute(gc, {"x": self.X[:, :4], "w1__A": A, "w1__B": B,
                                   "w2": _weights((4, 4), seed=5)})["y"]
        self.assertEqual(got.shape, ref.shape)


class TestInt8Quantization(unittest.TestCase):
    def test_per_column_scale_shape_and_dtype(self):
        W = _weights((16, 8), seed=6)
        base, scale = quantize_int8(W)
        self.assertEqual(base.dtype, np.int8)
        self.assertEqual(base.shape, (16, 8))
        self.assertEqual(scale.shape, (8,))
        self.assertEqual(scale.dtype, np.float32)

    def test_residual_within_half_step(self):
        W = _weights((64, 32), seed=7)
        base, scale = quantize_int8(W)
        deq = base.astype(np.float32) * scale[None, :]
        bound = np.max(scale / 2.0) * 1.01
        self.assertLessEqual(float(np.max(np.abs(W - deq))), bound)

    def test_int8_range_is_used(self):
        W = _weights((32, 16), seed=8)
        base, _ = quantize_int8(W)
        self.assertLessEqual(int(base.min()), -126)
        self.assertGreaterEqual(int(base.max()), 126)

    def test_all_zero_column_does_not_divide_by_zero(self):
        W = np.zeros((4, 4), np.float32)
        W[:, 0] = 1.0
        base, scale = quantize_int8(W)
        self.assertEqual(scale[0], 1.0 / 127.0)
        self.assertTrue(np.all(scale[1:] == 1.0))
        np.testing.assert_allclose(base.astype(np.float32) * scale, W, atol=1e-6)

    def test_rejects_non_2d_and_wrong_axis(self):
        with self.assertRaises(ValueError):
            quantize_int8(np.ones(4, np.float32))
        with self.assertRaises(ValueError):
            quantize_int8(np.ones((4, 4), np.float32), axis=0)


class TestQloraRewrite(unittest.TestCase):
    def setUp(self):
        self.g = _mlp()
        self.W = _weights((48, 32), seed=9)
        self.X = np.random.default_rng(10).standard_normal((8, 48)).astype(np.float32)
        self.DY = np.random.default_rng(11).standard_normal((8, 32)).astype(np.float32)

    def test_plan_picks_qlora_when_keep_base(self):
        plan = plan_param_compression(self.g, ranks={"w": 6}, keep_base={"w": True})
        self.assertEqual(plan["w"], ("qlora", 6))

    def test_base_is_declared_int8_in_ir(self):
        gc = ParamCompression({"w": ("qlora", 6)}).run(self.g)
        base = next(v for v in gc.inputs if v.name == "w__base")
        self.assertEqual(base.type.dtype.value, "i8")
        self.assertEqual(compressed_inputs(gc), {"w": "qlora"})

    def test_adapter_fits_the_quantization_residual(self):
        """qlora 的数值成立靠"适配器学量化丢掉的部分"，不是再拟合一次 W。"""
        cm = compress_for_training(self.g, {"w": self.W}, ranks={"w": 6},
                                   keep_base={"w": True}, data_inputs=["x"])
        base, scale = quantize_int8(self.W)
        residual = self.W - base.astype(np.float32) * scale[None, :]
        fitted = cm.trainable["w__A"] @ cm.trainable["w__B"]
        err = float(np.max(np.abs(residual - fitted)))
        self.assertLessEqual(err, float(np.max(scale / 2.0)) * 1.5)

    def test_double_fitting_would_be_worse(self):
        """反例判据：若适配器直接拟合 W（而非残差），误差必须明显更差。

        这一条钉住 qlora 的实现要点——底已经表示过 W 的主体，适配器再拟合
        一次 W 会双重计数。没有这条，"用残差 SVD"容易被后人当成可有可无的细节。
        """
        cm = compress_for_training(self.g, {"w": self.W}, ranks={"w": 6},
                                   keep_base={"w": True}, data_inputs=["x"])
        base, scale = quantize_int8(self.W)
        residual = self.W - base.astype(np.float32) * scale[None, :]
        good = cm.trainable["w__A"] @ cm.trainable["w__B"]
        A_bad, B_bad = low_rank_factors(self.W, 6)
        naive = A_bad @ B_bad
        e_good = float(np.max(np.abs(residual - good)))
        e_naive = float(np.max(np.abs(residual - naive)))
        self.assertLess(e_good, e_naive)

    def test_frozen_inputs_get_no_gradient(self):
        cm = compress_for_training(self.g, {"w": self.W}, ranks={"w": 6},
                                   keep_base={"w": True}, data_inputs=["x"])
        bwd = backward(cm.graph, wrt=[cm.graph.find("w__A"), cm.graph.find("w__B")])
        res = BACKEND.execute(bwd, {"x": self.X, **cm.frozen, **cm.trainable,
                                    "dy": self.DY})
        self.assertEqual(sorted(res), ["dw__A", "dw__B"])
        self.assertTrue(all(np.isfinite(v).all() for v in res.values()))

    def test_forward_matches_dense_within_combined_bound(self):
        cm = compress_for_training(self.g, {"w": self.W}, ranks={"w": 6},
                                   keep_base={"w": True}, data_inputs=["x"])
        ref = BACKEND.execute(self.g, {"x": self.X, "w": self.W})["y"]
        got = BACKEND.execute(cm.graph, {"x": self.X, **cm.frozen, **cm.trainable})["y"]
        base, scale = quantize_int8(self.W)
        s = np.linalg.svd(self.W - base.astype(np.float32) * scale[None, :],
                          compute_uv=False)
        trunc = float(np.sqrt(np.sum(s[6:] ** 2)))
        bound = np.linalg.norm(self.X, 2) * (float(np.max(scale / 2)) + trunc) * 1.05
        self.assertLessEqual(float(np.max(np.abs(ref - got))), bound)

    def test_batch_feed_rejects_overwriting_frozen(self):
        cm = compress_for_training(self.g, {"w": self.W}, ranks={"w": 6},
                                   keep_base={"w": True}, data_inputs=["x"])
        with self.assertRaises(KeyError):
            cm.feed({"w__base": np.zeros((48, 32), np.int8)})

    def test_weights_must_cover_uncompressed_float_inputs(self):
        g = Graph("two")
        x = g.input("x", f32[8, 4])
        w1 = g.input("w1", f32[4, 4])
        w2 = g.input("w2", f32[4, 4])
        g.output(g.add(g.matmul(x, w1), g.matmul(x, w2)), "y")
        with self.assertRaises(KeyError):
            compress_for_training(g, {"w1": _weights((4, 4))}, ranks={"w1": 2},
                                  data_inputs=["x"])

    def test_uncompressed_param_stays_trainable(self):
        g = Graph("two")
        x = g.input("x", f32[8, 4])
        w1 = g.input("w1", f32[4, 4])
        w2 = g.input("w2", f32[4, 4])
        g.output(g.add(g.matmul(x, w1), g.matmul(x, w2)), "y")
        W1, W2 = _weights((4, 4), seed=12), _weights((4, 4), seed=13)
        cm = compress_for_training(g, {"w1": W1, "w2": W2}, ranks={"w1": 2},
                                   data_inputs=["x"])
        self.assertIn("w2", cm.trainable)          # 没压的原样可训练
        self.assertNotIn("w1", cm.trainable)
        np.testing.assert_array_equal(cm.trainable["w2"], W2)


class TestMemoryAccounting(unittest.TestCase):
    def setUp(self):
        self.g = _mlp()
        self.W = _weights((48, 32), seed=14)
        self.cm = compress_for_training(self.g, {"w": self.W}, ranks={"w": 6},
                                        keep_base={"w": True}, data_inputs=["x"])

    def test_param_bytes_shrink(self):
        rep = memory_report({"w": self.W}, self.cm)
        self.assertLess(rep["compressed_bytes"], rep["original_bytes"])
        self.assertGreater(rep["ratio"], 1.0)

    def test_int8_base_is_a_quarter_of_dense(self):
        self.assertEqual(self.cm.frozen["w__base"].nbytes, self.W.nbytes // 4)

    def test_adam_slots_shrink_with_trainable_count(self):
        full = Adam({"w": self.W.copy()}, lr=1e-3)
        full.step({"w": np.ones_like(self.W)})
        store = ParameterStore(self.cm.trainable)
        cmp_opt = Adam(store.arrays(), lr=1e-3)
        cmp_opt.step({k: np.ones_like(v) for k, v in store.arrays().items()})
        size = lambda o: sum(a.nbytes for s in o.state_dict()["slots"].values() for a in s.values())
        self.assertLess(size(cmp_opt), size(full))

    def test_dense_free_check_passes_and_catches_leaks(self):
        store = ParameterStore(self.cm.trainable)
        assert_dense_free(store, self.cm.frozen, {"w": self.W.shape})
        leaked = dict(self.cm.trainable)
        leaked["w"] = self.W
        with self.assertRaises(AssertionError):
            assert_dense_free(ParameterStore(leaked), self.cm.frozen, {"w": self.W.shape})

    def test_dense_free_check_catches_frozen_in_store(self):
        """冻结量混进可训练表必须响：那等于"冻结"这个词没兑现。"""
        leaked = dict(self.cm.trainable)
        leaked["w__scale"] = self.cm.frozen["w__scale"].copy()
        with self.assertRaisesRegex(AssertionError, "冻结量"):
            assert_dense_free(ParameterStore(leaked), self.cm.frozen, {"w": self.W.shape})

    def test_dense_free_check_ignores_shape_coincidence(self):
        """同形不同名的合法参数不该被当成泄漏（按名字判，不按形状猜）。

        LM 图里 ``l0_A_raw`` 与被压的 ``l0_B_w`` 恰好同形，按形状查会误报。
        """
        other = np.ones((48, 32), np.float32)   # 与被压参数同形的合法参数
        table = {"w__A": self.cm.trainable["w__A"], "w__B": self.cm.trainable["w__B"],
                 "some_other_weight": other}
        assert_dense_free(ParameterStore(table), self.cm.frozen, {"w": self.W.shape})

    def test_frozen_base_must_be_int8(self):
        bad = dict(self.cm.frozen)
        bad["w__base"] = bad["w__base"].astype(np.float32)
        with self.assertRaises(AssertionError):
            assert_dense_free(ParameterStore(self.cm.trainable), bad, {"w": self.W.shape})


class TestLazyCorpusMemory(unittest.TestCase):
    """惰性批次的峰值必须远小于全量 one-hot——这是"不解压也能训练"的数字。"""

    def _corpus(self, rows=64, length=64, vocab=258):
        rng = np.random.default_rng(0)
        seqs = [list(int(b) for b in rng.integers(0, vocab, size=length))
                for _ in range(rows)]
        return TokenCorpus(seqs, vocab)

    def test_lazy_peak_is_order_of_one_batch(self):
        # 128 行 × 511 位 × V=258：全量 one-hot 约 8.3MB，单批 260KB，
        # 差 32 倍。判据要的是"批次量级、不是全量量级"这个 8 倍余量；
        # 语料只有几十行时"全量/8"会退化成"两批"，判据失去意义。
        corpus = self._corpus(rows=128, length=512)
        batch, steps, V = 4, 511, 258
        one_batch = batch * steps * V * 4
        whole = corpus.n_sequences * steps * V * 4
        self.assertGreater(whole / one_batch, 16)
        with peak_bytes() as box:
            for mb in corpus.iter_minibatches(batch, window=steps, labels="onehot"):
                del mb
        self.assertLess(box["peak"], whole / 8,
                        f"峰值 {box['peak']/1024:.0f}KB 已接近全量物化")
        self.assertLess(box["peak"], 6 * one_batch,
                        f"峰值应停在批次量级（单批 {one_batch/1024:.0f}KB）")

    def test_index_labels_never_materialize_onehot(self):
        corpus = self._corpus(rows=16, length=64)
        with peak_bytes() as box:
            for mb in corpus.iter_minibatches(4, window=63, labels="index"):
                self.assertNotEqual(mb["targets"].ndim, 3)
                del mb
        # index 口径一批只有一个 (B,T) i32，量级比 one-hot 小两个数量级
        self.assertLess(box["peak"], 4 * 63 * 258 * 4 * 4)


class TestCompressedLmEndToEnd(unittest.TestCase):
    """压缩态小 LM 在 π 上端到端训练：收敛 + 冻结不漂 + 无稠密副本。"""

    def _build(self, keep_base):
        # d_model=24：扫过 4 个秩 × 2 个 d_model × 5 个初始化后选的**稳定档**。
        # d_model=16 上出现过"loss 0.077 但贪心续写只有 1/10"，追查后发现是
        # 单一 seed 的巧合（同配置另 4 个 seed 全部 10/10），不是 qlora 的
        # 固有容量代价——所以这里选稳定档是为了测试不 flaky，不是为了
        # 掩盖什么问题。那条巧合本身的价值记在
        # test_lowrank_capacity_grows_with_rank 的说明里。
        spec = LMSpec(vocab_size=10, d_model=24, num_layers=1, d_state=8, d_conv=4,
                      steps=24, batch=4, tie_weights=True)
        g = Graph("lm")
        inputs, logits = build_language_model(g, spec, with_reset=True)
        t_in = g.input("targets", i32[4, 24])
        m_in = g.input("mask", f32[4, 24])
        g.output(masked_lm_loss_indexed(g, logits, t_in, m_in), "loss")

        rng = np.random.default_rng(5)
        data = {"ids", "targets", "mask", "reset"}
        weights = {
            v.name: (rng.standard_normal(tuple(int(d) for d in v.type.shape)) * 0.3).astype(np.float32)
            for v in g.inputs
            if v.name not in data and v.type.dtype.value == "f32"
        }
        matmul_w = {
            a.name for node in g.nodes if node.op.name == "matmul"
            for a in node.inputs
            if a.producer is None and len(a.type.shape) == 2
        }
        ranks = {k: min(4, min(weights[k].shape))
                 for k in weights if k in matmul_w and min(weights[k].shape) >= 2}
        cm = compress_for_training(g, weights, ranks=ranks,
                                   keep_base={k: keep_base for k in ranks},
                                   data_inputs=data)
        return cm, weights, ranks, spec

    def _corpus(self):
        seqs = [PI_TOKENS[i:i + 25] for i in range(0, len(PI_TOKENS) - 24, 25)]
        seqs = seqs[: (len(seqs) // 4) * 4]
        return TokenCorpus(seqs, 10)

    def _train(self, cm, epochs=20):
        store = ParameterStore.from_graph(cm.graph, exclude=cm.exclude, seed=1)
        for k, v in cm.trainable.items():
            store[k][...] = v
        trainer = Trainer(cm.graph, store, Adam(store.arrays(), lr=0.05), output="loss")
        corpus = self._corpus()
        losses = []
        frozen_before = {k: v.copy() for k, v in cm.frozen.items()}
        for _ in range(epochs):
            for batch in corpus.iter_minibatches(4, window=24, labels="index"):
                losses.append(float(trainer.step(cm.feed(batch))))
        return losses, store, frozen_before, corpus

    def test_qlora_converges(self):
        cm, _, _, _ = self._build(keep_base=True)
        losses, store, frozen_before, _ = self._train(cm)
        first, last = float(np.mean(losses[:4])), float(np.mean(losses[-4:]))
        self.assertLess(last, first * 0.5, f"loss {first:.3f} -> {last:.3f}")

    def test_low_rank_converges(self):
        cm, _, _, _ = self._build(keep_base=False)
        losses, *_ = self._train(cm)
        first, last = float(np.mean(losses[:4])), float(np.mean(losses[-4:]))
        self.assertLess(last, first * 0.5, f"loss {first:.3f} -> {last:.3f}")

    def test_frozen_bases_never_move(self):
        cm, _, ranks, _ = self._build(keep_base=True)
        losses, store, frozen_before, corpus = self._train(cm, epochs=6)
        for k, v in cm.frozen.items():
            np.testing.assert_array_equal(v, frozen_before[k],
                                          err_msg=f"冻结量 {k} 在训练中被改写")
        self.assertNotIn(any(n.endswith(("__base", "__scale")) for n in store.names),
                         [True], "冻结量进了参数表")

    def test_no_dense_copy_of_compressed_weights(self):
        cm, weights, ranks, _ = self._build(keep_base=True)
        store = ParameterStore(cm.trainable)
        assert_dense_free(store, cm.frozen, {k: weights[k].shape for k in ranks})

    def test_memory_report_shows_real_reduction(self):
        cm, weights, _, _ = self._build(keep_base=True)
        rep = memory_report(weights, cm)
        self.assertLess(rep["compressed_bytes"], rep["original_bytes"])

    def test_compressed_weights_rebuild_to_dense_for_inference(self):
        """推理引擎吃的是稠密 f32 权重：``dequant + A@B`` 重建必须落回原值附近。

        训练驻留的是压缩形态，推理走 ``generate`` 的稠密接口——两个世界靠
        ``dense_weights`` 这一步对接。误差界用与训练侧同一条判据（量化
        半格点 + 截断奇异值）。
        """
        cm, weights, ranks, _ = self._build(keep_base=True)
        dense = cm.dense_weights(cm.trainable)   # 未训练：对账用初始因子
        for k in ranks:
            W = weights[k]
            self.assertEqual(dense[k].shape, W.shape)
            self.assertEqual(dense[k].dtype, np.float32)
            base = cm.frozen[f"{k}__base"].astype(np.float32)
            scale = cm.frozen[f"{k}__scale"]
            s = np.linalg.svd(W - base * scale[None, :], compute_uv=False)
            bound = float(np.max(scale / 2.0)) + float(np.sqrt(np.sum(s[ranks[k]:] ** 2)))
            self.assertLessEqual(float(np.max(np.abs(W - dense[k]))), bound * 1.5,
                                 f"参数 {k} 的重建误差越界")
        # 未压缩的参数必须也在表里，且就是原值
        self.assertIn("embed", dense)
        np.testing.assert_array_equal(dense["embed"], weights["embed"])

    def test_dense_weights_refuses_to_silently_use_initial_factors(self):
        """``dense_weights(None)`` 必须报错，而不是回落初始分解。

        "忘了传训练结果"的后果是一份看起来合理、实际从没训练过的权重；
        让它变成显式 TypeError 是这条 API 存在的理由。
        """
        cm, *_ = self._build(keep_base=True)
        with self.assertRaises(TypeError):
            cm.dense_weights(None)

    def test_half_a_factor_pair_is_refused(self):
        """只剩 ``__B`` 没有 ``__A`` 时必须报错，不能把它当普通参数搬走。

        半套因子如果被当成普通参数原样放进稠密表，推理图会拿到一个形状
        完全不对的"w2__B"、同时缺真正的 w2——两个世界接错线里最安静的一种。
        """
        cm, _, _, _ = self._build(keep_base=True)
        broken = dict(cm.trainable)
        k0 = sorted(k for k in cm.policy)[0]
        broken.pop(f"{k0}__A")
        with self.assertRaisesRegex(KeyError, "配对的"):
            cm.dense_weights(broken)
        # 反过来（只剩 A）同样要响
        broken2 = dict(cm.trainable)
        broken2.pop(f"{k0}__B")
        with self.assertRaisesRegex(KeyError, "配对的"):
            cm.dense_weights(broken2)

    def test_generation_uses_trained_not_initial_weights(self):
        """判据的判据：重建出的稠密表必须反映训练，而不是初始分解。

        拿 ``store.arrays()`` 与拿 ``cm.trainable`` 重建出的被压权重必须
        **不同**（训练动了因子），而且只有前者能生成对的东西。这条专门
        防住"两个世界接错线"的实现错误——它比下一条生成测试更重要，
        因为生成测试可能靠运气过。
        """
        cm, weights, ranks, _ = self._build(keep_base=True)
        _, store, _, _ = self._train(cm, epochs=8)
        trained = cm.dense_weights(store.arrays())
        initial = cm.dense_weights(cm.trainable)
        k0 = sorted(ranks)[0]
        self.assertFalse(
            np.allclose(trained[k0], initial[k0]),
            "训练后的稠密重建与初始分解相同——说明 store 的更新没进重建",
        )

    def test_low_rank_generation_matches_plain_quality(self):
        """lowrank（整份 W 由因子可训练）必须与未压缩同档：10/10 续写。

        这条是"压缩不损失能力"的正证：低秩因子仍在训练全部容量。
        """
        from aetherflow.models.lm import generate

        cm, weights, ranks, spec = self._build(keep_base=False)
        losses, store, _, _ = self._train(cm, epochs=40)
        dense = cm.dense_weights(store.arrays())
        gen = list(generate(BACKEND, dense, [3, 1, 4, 1, 5], spec.with_batch(1),
                            max_new_tokens=10, eos_id=255))
        truth = PI_TOKENS[5:15]
        hits = sum(int(a == b) for a, b in zip(gen, truth))
        self.assertGreaterEqual(hits, 9, f"gen={gen} truth={truth}")

    def test_lowrank_capacity_grows_with_rank(self):
        """lowrank 的可训练容量必须随秩单调增长——这是唯一稳定的规律。

        实测（d_model=16，同一份数据、同样 40 epoch，秩是唯一的变量）：

        | 秩 | 末尾 loss | 贪心续写 |
        |---|---|---|
        | 1 | 1.57 | 0/10 |
        | 2 | 1.01 | 5/10 |
        | 4 | 0.12 | 10/10 |

        判据钉住两端（秩 1 明显学不会、秩 4 学全），中间点不钉：它是
        趋势而不是可复现的定值。

        **为什么不给 qlora 立同样形状的判据**：扫了 4 个秩 × 2 个 d_model
        × 5 个初始化之后，qlora 的"生成准确率 vs 秩"没有单调关系——
        d=16 秩 1/2/8 都 10/10，秩 4 在 seed 5 上只有 1/10（loss 0.077，
        另外 4 个 seed 同配置都是 10/10）。那是**单次初始化的巧合**，
        把它写成断言等于把巧合当规律，比不测更糟。真正留下的观察是：
        **贪心 rollout 对权重扰动比 teacher-forced loss 敏感得多**——
        loss 0.077 照样能一步走偏后一路跑飞。这条只在文档里记着，没有
        做成断言（它需要多种子统计才有资格当判据）。留着下面这条 lowrank
        判据，是因为它的方向在全因子可训练的路径上跨 d_model 都成立。
        """
        from aetherflow.models.lm import generate

        spec = LMSpec(vocab_size=10, d_model=16, num_layers=1, d_state=8, d_conv=4,
                      steps=24, batch=4, tie_weights=True)
        g = Graph("lm")
        _, logits = build_language_model(g, spec, with_reset=True)
        t_in = g.input("targets", i32[4, 24])
        m_in = g.input("mask", f32[4, 24])
        g.output(masked_lm_loss_indexed(g, logits, t_in, m_in), "loss")
        rng = np.random.default_rng(5)
        data = {"ids", "targets", "mask", "reset"}
        weights = {
            v.name: (rng.standard_normal(tuple(int(d) for d in v.type.shape)) * 0.3).astype(np.float32)
            for v in g.inputs if v.name not in data and v.type.dtype.value == "f32"
        }
        mm = {a.name for node in g.nodes if node.op.name == "matmul"
              for a in node.inputs if a.producer is None and len(a.type.shape) == 2}
        corpus = self._corpus()

        def train_at(rank):
            ranks = {k: rank for k in weights if k in mm and min(weights[k].shape) > rank}
            cm = compress_for_training(g, weights, ranks=ranks, data_inputs=data)
            store = ParameterStore.from_graph(cm.graph, exclude=cm.exclude, seed=1)
            for k, v in cm.trainable.items():
                store[k][...] = v
            trainer = Trainer(cm.graph, store, Adam(store.arrays(), lr=0.05), output="loss")
            losses = []
            for _ in range(40):
                for batch in corpus.iter_minibatches(4, window=24, labels="index"):
                    losses.append(float(trainer.step(cm.feed(batch))))
            gen = list(generate(BACKEND, cm.dense_weights(store.arrays()),
                                [3, 1, 4, 1, 5], spec.with_batch(1),
                                max_new_tokens=10, eos_id=255))
            truth = PI_TOKENS[5:15]
            return float(np.mean(losses[-4:])), sum(int(a == b) for a, b in zip(gen, truth))

        loss_small, hits_small = train_at(1)
        loss_big, hits_big = train_at(4)
        self.assertGreater(loss_small, loss_big * 3,
                           f"秩 1 的 loss {loss_small:.3f} 应当远高于秩 4 {loss_big:.3f}")
        self.assertLessEqual(hits_small, 3, f"秩 1 不该学会续写：{hits_small}/10")
        self.assertGreaterEqual(hits_big, 9, f"秩 4 应当学全：{hits_big}/10")

    def test_generation_from_trained_compressed_state(self):
        """端到端最后一环：压缩训练出来的权重，重建后交给 generate 能续写 π。

        训练全程驻留压缩形态（int8 底 + 因子）；推理引擎 ``generate`` 需要
        batch=1 的稠密参数。判据用准确率而不是逐位相等：贪心解码对 logits
        的微小差异敏感，但收敛到 π 的模型应该能大段续写正确。
        """
        from aetherflow.models.lm import generate

        cm, weights, ranks, spec = self._build(keep_base=True)
        losses, store, _, _ = self._train(cm, epochs=40)
        dense_params = cm.dense_weights(store.arrays())

        missing = set(weights) - set(dense_params)
        self.assertEqual(missing, set(), f"重建后仍缺参数 {sorted(missing)}")

        gen = list(generate(BACKEND, dense_params, [3, 1, 4, 1, 5],
                            spec.with_batch(1), max_new_tokens=10,
                            # 词表 10 的数字语料没有 EOS：spec 默认 eos_id=1
                            # 就是数字 "1"。None 会回落到它（models/lm 的
                            # stop 口径），所以关掉早停必须传一个不出现的 id。
                            # 不关的话"没学好"与"预测到 1 就停"两种失败混成
                            # 一件事——实测 qlora 在 d=16 只吐出 2 个 token，
                            # 看起来像崩了，其实是撞了 EOS。）
                            eos_id=255))
        truth = PI_TOKENS[5:15]
        hits = sum(int(a == b) for a, b in zip(gen, truth))
        first = float(np.mean(losses[:4]))
        last = float(np.mean(losses[-4:]))
        self.assertGreaterEqual(hits, 6,
            f"续写准确率过低（{hits}/10）: gen={list(gen)} truth={truth}, "
            f"loss {first:.3f}->{last:.3f}")


if __name__ == "__main__":
    unittest.main()
