# 第 14 课：ConvNeXt

## 这一代要解决的问题

把 ResNet 按 Transformer 的习惯逐项改掉之后，卷积网络还强不强。ConvNeXt 使用大核深度卷积、LayerNorm、GELU、倒瓶颈，以及更少的激活。

## 知识要点

一个 ConvNeXt block 的数据流：

```text
7x7 depthwise conv
-> 换成 (N, H, W, C)
-> LayerNorm
-> Linear 放大 4 倍
-> GELU
-> Linear 压回
-> 换回 (N, C, H, W)
-> 残差相加
```

- 7×7 且 `groups=dim`，就是第 11 课那种 depthwise 卷积，只是核更大。
- 通道混合发生在后面的 `Linear`，对应 MobileNet 里混合通道的 1×1。
- LayerNorm 在 PyTorch 里默认作用在最后一维，所以要先把通道挪到最后，算完再挪回来。
- stem 使用 2×2 stride 2，把 32×32 收成 16×16，模仿 ViT 的 patchify，只是 patch 更小。
- 教学网络只有两个 stage，每个 stage 两个块。完整 ConvNeXt 有四个 stage。

## 结构与形状表

输入 `(N, 3, 32, 32)`。

| 层 | 输出 |
| --- | --- |
| stem，k=2，s=2，64 通道 | `(N, 64, 16, 16)` |
| 两个 ConvNeXtBlock(64) | `(N, 64, 16, 16)` |
| 下采样 k=2，s=2，128 通道 | `(N, 128, 8, 8)` |
| 两个 ConvNeXtBlock(128) | `(N, 128, 8, 8)` |
| 全局池化 + Linear | `(N, 10)` |

block 内部的 LayerNorm 输入是 `(N, 16, 16, 64)` 这种 NHWC。形状日志里如果看到最后一维是通道，说明 permute 生效了。

## 对应例程导读

[code/lesson14_convnext.py](../code/lesson14_convnext.py) 的 `ConvNeXtBlock`。

`forward` 先保存 `shortcut = x`，此时 `x` 仍是 NCHW。depthwise 不改变这个布局。两次 `permute` 必须成对：`(0, 2, 3, 1)` 出去，`(0, 3, 1, 2)` 回来。回来之后才能和 `shortcut` 相加。

## 动手实验

```bash
python code/lesson14_convnext.py
python code/lesson14_convnext.py --mode train --epochs 30
```

在日志里指出哪一层是 7×7 depthwise，哪一层把通道放大 4 倍。结果写入 [results.md](../results.md)。

## 思考题

1. 7×7 且 `groups=dim`，和第 11 课的 depthwise 是不是同一种卷积？
2. 如果少写一次 permute，LayerNorm 会作用在哪个维度上？
3. stem 的 2×2 stride 2，和第 1 课公式算出来的输出高宽是否一致？
