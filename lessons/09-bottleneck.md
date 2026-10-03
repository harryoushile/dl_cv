# 第 9 课：Bottleneck 与去掉捷径

## 这一代要解决的问题

把 BasicBlock 一直堆到 50 层，中间的 3×3 太贵。ResNet-50 把每个块换成瓶颈：1×1 降通道，3×3 在窄通道上算，1×1 再升到 4 倍。本课还要亲眼看到：同样的层数，去掉捷径会更难训。

## 知识要点

- `Bottleneck.expansion = 4`。中间通道是 `mid_ch`，出口通道是 `mid_ch * 4`。
- 顺序是 1×1、3×3、1×1，前两次卷积后 ReLU，第三次卷积后先和捷径相加，再 ReLU。
- 形状检查使用 `Bottleneck(256, 64, stride=2)`：入口 256，中间 64，出口 `64 * 4 = 256`，高宽减半。
- 去掉捷径时，`BasicBlock` 仍计算投影卷积，只是 `forward` 不把捷径加回去。这样参数量不变，对比的是「有没有相加」，不是「参数变少了」。
- 这是优化上的对照。训练误差如果更高，说明网络更难拟合，而不只是测试时过拟合。

## 结构与形状表

Bottleneck 检查输入 `(2, 256, 32, 32)`：

| 层 | 输出 |
| --- | --- |
| 1×1，256→64 | `(2, 64, 32, 32)` |
| 3×3，stride 2 | `(2, 64, 16, 16)` |
| 1×1，64→256 | `(2, 256, 16, 16)` |
| 捷径 1×1，stride 2 | `(2, 256, 16, 16)` |
| 相加后 ReLU | `(2, 256, 16, 16)` |

完整分类网络仍是上一课的 ResNet-18。去掉捷径后，最终 logits 仍是 `(2, 10)`。

## 对应例程导读

代码仍在 [code/lesson08_resnet.py](../code/lesson08_resnet.py)。

`Bottleneck.forward` 里 `out + self.shortcut(x)` 要求两条路已经对齐。`ResNet18CIFAR(use_shortcut=False)` 由命令行 `--no-shortcut` 打开。

形状模式会打印两段：ResNet-18 的逐层形状，以及一行 `bottleneck (2, 256, 16, 16)`。

## 动手实验

```bash
python code/lesson08_resnet.py
python code/lesson08_resnet.py --mode train --epochs 30 --no-shortcut
```

和上一课同为 30 个 epoch 的 ResNet-18 比较验证准确率。在 [results.md](../results.md) 写一句结论：去掉相加之后训练变得更容易还是更难。

## 思考题

1. 中间通道是 64、expansion 是 4 时，出口通道为什么是 256？
2. 为什么消融要保留捷径上的 1×1 参数，只取消相加？
3. 若把 ReLU 放到相加之前，和现在的 `forward` 差在哪一行？
