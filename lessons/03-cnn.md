# 第 3 课：卷积、池化、感受野

## 这一代要解决的问题

全连接把每个像素和所有输出连在一起，相邻像素的关系要靠数据自己学。卷积只看局部，并且在整张图上共享同一组权重。一个核学一种图案。

## 知识要点

- `Conv2d` 权重形状是 `(out_channels, in_channels, k, k)`。
- 多个核就是多个输出通道。
- 最大池化把特征图缩小，下一层因此看到更大的原图范围，这个范围叫感受野。
- `padding=1`、`kernel=3`、`stride=1` 时高宽不变。
- `x.view(x.size(0), -1)` 放在全连接之前，把特征图拉直。

## 结构与形状表

输入 `(N, 1, 28, 28)`。

| 层 | 输出 |
| --- | --- |
| Conv 1→8，k=3，p=1 | `(N, 8, 28, 28)` |
| MaxPool 2×2 | `(N, 8, 14, 14)` |
| Conv 8→16，k=3，p=1 | `(N, 16, 14, 14)` |
| MaxPool 2×2 | `(N, 16, 7, 7)` |
| Flatten | `(N, 784)` |
| Linear | `(N, 10)` |

第一层卷积参数量是 `8 * 1 * 3 * 3 + 8`。8 个偏置来自每个输出通道一个。

## 对应例程导读

[code/lesson03_tinycnn.py](../code/lesson03_tinycnn.py) 把特征提取放进 `self.features`，分类放进 `self.classifier`。

`features` 是 `nn.Sequential`。形状日志里会看到 `features.0`、`features.1` 这种名字，数字是 Sequential 里的顺序。池化没有参数，但仍然会出现在日志里，因为它改变高宽。

如果把第一层 `padding` 改成 0，拉直后的长度不再是 784，`Linear` 会报错。这是本课最有价值的错误。

## 动手实验

```bash
python code/lesson03_tinycnn.py
python code/lesson03_tinycnn.py --mode train --epochs 5
```

和上一课比参数量、验证准确率。再在交互环境打印 `model.features[0].weight.shape`，对照 `(8, 1, 3, 3)`。

## 思考题

1. 第二层卷积的 `in_channels` 为什么必须是 8？
2. 池化层的参数量为什么是 0？
3. 把两层卷积的输出通道都乘 2，全连接的输入维度怎么变？
