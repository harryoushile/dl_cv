# 第 5 课：AlexNet

## 这一代要解决的问题

2012 年 AlexNet 用更深的卷积、ReLU、Dropout 和 GPU，在 1000 类 ImageNet 上超过传统手工特征。本课学习这些部件，并在 CIFAR-10 上写一个能训的教学版。

## 知识要点

- ReLU 把负数变成 0，正数原样通过，深层里比 sigmoid 更不容易饱和。
- Dropout 在训练时随机把一部分激活置 0，减轻全连接过拟合。`model.eval()` 时它自动关掉，所以测试代码不用手动关闭 Dropout。
- ImageNet 原版输入 227×227：11×11 stride 4 的 96 通道卷积，然后 5×5 的 256 通道，三段 3×3（384、384、256），三次最大池化，再接 4096、4096、1000 的全连接。参数大约六千万，大部分在全连接里。
- CIFAR-10 是 32×32、10 类。教学版保留「下采样、3×3 堆叠、Dropout 全连接」，改掉三处：输入尺寸、第一层卷积、全连接宽度。

## 结构与形状表

教学版输入 `(N, 3, 32, 32)`。

| 层 | 输出 |
| --- | --- |
| Conv 3→64，k=3，p=1 | `(N, 64, 32, 32)` |
| MaxPool | `(N, 64, 16, 16)` |
| Conv 64→192，k=3，p=1，再 Pool | `(N, 192, 8, 8)` |
| 三段 3×3：384、256、256 | `(N, 256, 8, 8)` |
| MaxPool | `(N, 256, 4, 4)` |
| Dropout + Linear 4096→1024→1024→10 | `(N, 10)` |

`256 * 4 * 4 = 4096`，所以第一个全连接写成 `Linear(4096, 1024)`。

## 对应例程导读

[code/lesson05_alexnet.py](../code/lesson05_alexnet.py) 的 `AlexNetCIFAR`。

文件顶部的说明列出了和原版的差异。第一层是 3×3 stride 1，原版是 11×11 stride 4。全连接是 1024 宽，原版是 4096，最后一层是 10 类而不是 1000。

`Dropout(0.5)` 写在 `classifier` 里。形状检查走 `eval()`，Dropout 此时是恒等映射，日志里输入输出形状相同。训练时它才随机丢弃。

## 动手实验

```bash
python code/lesson05_alexnet.py
python code/lesson05_alexnet.py --mode train --epochs 20
```

报告里写明和 ImageNet 原版差在哪三处。把参数量和准确率填进 [results.md](../results.md)。

## 思考题

1. `model.train()` 和 `model.eval()` 对 Dropout 分别做什么？
2. 原版参数大部分在全连接。把本课第一个 `Linear` 的 1024 改成 4096，参数量大约变成几倍？只估算这一层即可。
3. 为什么 CIFAR 上不直接使用 11×11 stride 4？32×32 的图经过这一层后高宽还剩多少？
