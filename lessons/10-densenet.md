# 第 10 课：DenseNet

## 这一代要解决的问题

ResNet 用加法把输入接到输出，通道数不变。DenseNet 把这一层和前面每一层在通道维拼接，后面的层可以直接读到早期特征。

## 知识要点

- 一个 dense layer 新产出的通道数叫 growth rate，记为 `k`。入口 `C` 通道，拼接后是 `C + k`。
- 通道会随层数线性增加，所以块和块之间用 Transition：1×1 把通道压小，平均池化把高宽减半。
- 拼接和相加不同。拼接要求高宽相同，通道相加；相加要求四个维度都相同。
- 例程的 growth 是 12，每个 dense block 有 4 层。这是教学宽度，不是 ImageNet DenseNet-121 的完整配置。

## 结构与形状表

stem 把 3 通道变成 32，空间仍是 32×32。

| 位置 | 通道怎么来 | 通道数 |
| --- | --- | --- |
| stem | 3→32 | 32 |
| block1 四层 | 32+12+12+12+12 | 80 |
| transition | 80→40，高宽 32→16 | 40 |
| block2 四层 | 40+12×4 | 88 |
| 池化 + Linear | 88→10 | 10 |

block1 出口的高宽仍是 32，因为 3×3 卷积 `padding=1`。变小只发生在 Transition 的 `AvgPool2d(2)`。

## 对应例程导读

[code/lesson10_densenet.py](../code/lesson10_densenet.py)。

`DenseLayer.forward` 先做 BN-ReLU-Conv，再 `torch.cat([x, out], dim=1)`。`MiniDenseNet` 用列表推导生成四层，第 `i` 层的输入通道写成 `ch + i * growth`。这个式子必须和拼接顺序一致：第 0 层看 32，第 1 层看 44，以此类推。

## 动手实验

```bash
python code/lesson10_densenet.py
python code/lesson10_densenet.py --mode train --epochs 30
```

在形状日志里核对 block1 四个卷积的输出通道都是 12，而模块整体的输出通道逐层增加。把结果填进 [results.md](../results.md)。

## 思考题

1. 4 层、growth=12、块入口 32，块出口通道为什么是 80？
2. 如果忘记 Transition、把两个 4 层块直接接上，第二块第一层的 `in_ch` 应该写成多少？
3. DenseNet 的拼接和 ResNet 的相加，对通道数的影响有什么不同？
