# 第 11 课：MobileNetV2

## 这一代要解决的问题

前面的网络精度高，普通 3×3 卷积太贵，放到手机上不合适。MobileNet 用深度可分离卷积和倒残差减少计算。

## 知识要点

- 深度可分离卷积分两步：每个通道单独做 3×3（depthwise，`groups` 等于通道数），再用 1×1 混合通道（pointwise）。
- depthwise 卷积的权重形状是 `(hidden, 1, 3, 3)`，因为每个核只看 1 个输入通道。
- 倒残差和 ResNet 瓶颈相反：1×1 先把通道放大（这里是 6 倍），depthwise 3×3 在宽通道上提取空间特征，1×1 再压回去。压回去的那一层不加 ReLU，避免已经很少的通道再被截成 0。例程用 `ReLU6`，这是 MobileNet 的激活。
- 只有 stride 为 1 且输入输出通道相同，才能 `x + out`。stride 为 2 时高宽已经不同，残差必须关掉。

## 结构与形状表

输入 `(N, 3, 32, 32)`。

| 层 | 输出 |
| --- | --- |
| Conv 3→16，k=3，p=1 | `(N, 16, 32, 32)` |
| InvertedResidual 16→24，stride 2 | `(N, 24, 16, 16)` |
| InvertedResidual 24→24，stride 1 | `(N, 24, 16, 16)` |
| InvertedResidual 24→32，stride 2 | `(N, 32, 8, 8)` |
| InvertedResidual 32→32，stride 1 | `(N, 32, 8, 8)` |
| InvertedResidual 32→64，stride 2 | `(N, 64, 4, 4)` |
| 全局池化 + Linear | `(N, 10)` |

第一块里 expand=6，所以 16 通道先变成 96，depthwise 的 `groups` 是 96。

## 对应例程导读

[code/lesson11_mobilenet.py](../code/lesson11_mobilenet.py) 的 `InvertedResidual`。

`self.use_res = stride == 1 and in_ch == out_ch` 决定 `forward` 是否相加。放大分支在 `expand != 1` 时才加入。最后的 1×1 后面只有 BatchNorm，没有 ReLU。

形状日志里找到第一个 depthwise 卷积，输出通道等于它的输入通道，空间尺寸随 stride 变化。

## 动手实验

```bash
python code/lesson11_mobilenet.py
python code/lesson11_mobilenet.py --mode train --epochs 30
```

打印第一个 `groups` 不为 1 的卷积权重形状，确认第二维是 1。和 ResNet-18 比较参数量。结果写入 [results.md](../results.md)。

## 思考题

1. stride 为 2 时，为什么不能把输入和输出相加？
2. `groups=hidden` 时，权重为什么是 `(hidden, 1, 3, 3)` 而不是 `(hidden, hidden, 3, 3)`？
3. 倒残差中间宽、两头窄。ResNet 的 Bottleneck 中间是宽还是窄？
