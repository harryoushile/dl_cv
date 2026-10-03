# 第 8 课：ResNet 与 BasicBlock

## 这一代要解决的问题

VGG 加深到一定程度后，训练误差反而变大。这是优化困难：层要学的映射如果接近恒等，直接学恒等比学一个接近 0 的增量更难。残差把层的任务改成学习增量。

```text
y = F(x) + x
```

`F` 是两层卷积。捷径把 `x` 加回去。

## 知识要点

- 高宽和通道都不变时，捷径是 `nn.Identity()`，直接相加。
- stride 为 2 或通道变化时，捷径用 1×1 卷积把 `x` 投影到同一形状，再相加。
- 主路顺序是 Conv-BN-ReLU、Conv-BN，相加之后再 ReLU。卷积使用 `bias=False`，因为 BatchNorm 自己有偏移。
- ImageNet 的 ResNet 第一层是 7×7 stride 2，再接最大池化。CIFAR 只有 32×32，教学版用 3×3 stride 1 开头。这是 CIFAR 实验的常见写法，不是写错。
- 每个 stage 第一个块负责下采样和升通道，后面的块 stride 为 1。

## 结构与形状表

输入 `(N, 3, 32, 32)` 的 ResNet-18 教学版：

| 阶段 | 输出 |
| --- | --- |
| stem：Conv 3→64，k=3，p=1 | `(N, 64, 32, 32)` |
| layer1：两个 BasicBlock，64 | `(N, 64, 32, 32)` |
| layer2：第一个 stride 2，128 | `(N, 128, 16, 16)` |
| layer3：256 | `(N, 256, 8, 8)` |
| layer4：512 | `(N, 512, 4, 4)` |
| AdaptiveAvgPool + Linear | `(N, 10)` |

`layer2` 的第一个块里，主路和 1×1 捷径都必须是 `(N, 128, 16, 16)` 才能相加。

## 对应例程导读

[code/lesson08_resnet.py](../code/lesson08_resnet.py) 的 `BasicBlock` 和 `ResNet18CIFAR`。

`use_shortcut=False` 时不做相加，主路仍按 stride 改变形状，所以网络还能前向。第 9 课用这个开关做消融。

形状检查还会额外跑一个 `Bottleneck`，打印 `bottleneck (2, 256, 16, 16)`。Bottleneck 的讲解放在第 9 课，本课先确认 BasicBlock 的输出是 `(2, 10)`。

## 动手实验

```bash
python code/lesson08_resnet.py
python code/lesson08_resnet.py --mode train --epochs 30
```

在形状日志里找到 `layer2` 第一个块的捷径卷积，确认它的输出高宽是 16、通道是 128。把参数量和准确率填进 [results.md](../results.md)，留给下一课对比。

## 思考题

1. 什么条件下捷径必须用 1×1，什么条件下可以恒等？
2. 相加的两个张量，四个维度里哪一个不同都会报错？
3. 为什么 stem 在 CIFAR 上不用 7×7 stride 2？32 的边长马上会变成多少？
