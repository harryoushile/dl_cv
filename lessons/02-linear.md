# 第 2 课：自动求导与线性分类器

## 这一代要解决的问题

还没有卷积时，能不能把整张图拉成一条向量，直接分成 10 类。这个模型是后面所有网络的对照：它能跑通训练，但看不到局部图案。

## 知识要点

- `loss.backward()` 把梯度写进 `param.grad`。
- `optimizer.step()` 用梯度更新参数。
- `optimizer.zero_grad()` 清掉上一步的梯度，否则梯度会累加。
- 交叉熵损失接收的是没有做 softmax 的 logits，形状 `(N, 类别数)`。
- 子模块必须在 `__init__` 里赋给 `self`。写在 `forward` 里临时创建的层，优化器看不到。

MNIST 训练使用 Adam，学习率 `1e-3`，见 [common/train_mnist.py](../common/train_mnist.py)。

## 结构与形状表

输入 `(N, 1, 28, 28)`。

| 层 | 输出 |
| --- | --- |
| `view(N, -1)` | `(N, 784)` |
| `Linear(784, 10)` | `(N, 10)` |

参数量是 `784 * 10 + 10`。偏置每个输出类别一个。

## 对应例程导读

模型在 [code/lesson02_linear.py](../code/lesson02_linear.py) 的 `LinearMNIST`。

`forward` 里 `x.view(x.size(0), -1)` 用当前 batch 大小，而不是写死 128。这样形状检查的 batch=2 和训练的 batch=128 都能过。

`run_shapes` 会打印叶子模块的输入输出。这个网络只有一个 `fc`，所以形状日志很短。最后一行 `output` 应是 `(2, 10)`。

## 动手实验

```bash
python code/lesson02_linear.py
python code/lesson02_linear.py --mode train --epochs 3
```

把参数量和验证准确率填进 [results.md](../results.md)。这是第 3、4 课的基线。

## 思考题

1. 把 `view` 写成 `x.view(-1)`，batch 大于 1 时输出形状变成什么？
2. 为什么损失下降时，准确率不一定每一轮都上升？
3. 这个模型和「每个像素单独设一个权重连到 10 个类别」是同一件事吗？参数量怎么对应？
