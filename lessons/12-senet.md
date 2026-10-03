# 第 12 课：SENet 与通道注意力

## 这一代要解决的问题

一个卷积层输出很多通道，它们不一定同等重要。SE 模块给每个通道一个 0 到 1 的权重，再乘回特征图。

## 知识要点

- Squeeze：对每个通道做全局平均，`(N, C, H, W)` 变成 `(N, C)`。
- Excitation：两层全连接，中间缩小再放大，最后 Sigmoid，得到每个通道的权重。
- 权重 reshape 成 `(N, C, 1, 1)` 后和特征图相乘，高宽会被广播。
- `reduction=16` 表示中间层通道大约是 `C / 16`。通道很小时用 `max(C // 16, 4)`，避免中间层变成 0。
- 标准 SE-ResNet 把 SE 放在残差分支的第二个 BN 之后、和捷径相加之前。教学块为了少改第 8 课的 `BasicBlock`，把 SE 放在整个残差块输出之后。
- EfficientNet 的阅读点：它反复使用倒残差加 SE，并用一组比例同时增加深度、宽度和输入分辨率。本课不实现完整 EfficientNet。

## 结构与形状表

SE 本身不改变形状。输入 `(N, 64, 32, 32)` 时：

| 步骤 | 形状 |
| --- | --- |
| 全局平均 | `(N, 64)` |
| Linear 64→4，ReLU | `(N, 4)` |
| Linear 4→64，Sigmoid | `(N, 64)` |
| view | `(N, 64, 1, 1)` |
| 乘回特征图 | `(N, 64, 32, 32)` |

整个 SE-ResNet-18 的空间路径和第 8 课相同，最终 logits 是 `(N, 10)`。

## 对应例程导读

[code/lesson12_senet.py](../code/lesson12_senet.py)。

`SE.forward` 用 `x.mean(dim=(2, 3))` 做挤压。`SEBasicBlock` 先调用第 8 课的 `BasicBlock`，再把输出送进 `SE`。`SEResNet18CIFAR` 的 stage 划分和 `ResNet18CIFAR` 一致，只是块的类型换成 `SEBasicBlock`。

## 动手实验

```bash
python code/lesson12_senet.py
python code/lesson12_senet.py --mode train --epochs 30
```

训练之后，取一个 batch，单独调用某一层 `SE`，把 Sigmoid 之前或之后的 0 到 1 权重打印出来，看不同通道是否被压低。准确率填进 [results.md](../results.md)。

## 思考题

1. 全局平均之后为什么还要两层全连接，而不是直接用这个平均值当权重？
2. 权重形状是 `(N, C, 1, 1)`，它如何乘到 `(N, C, H, W)` 上？
3. 把 SE 挪到相加之前，需要改 `BasicBlock.forward` 的哪一步？
