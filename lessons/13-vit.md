# 第 13 课：Vision Transformer

## 这一代要解决的问题

把图像切成小块，当成一串 token，只用自注意力做分类，看卷积是不是必需的。

## 知识要点

- Patch embed 用一次卷积完成切块：核大小和 stride 都等于 patch。32×32、patch=4 时得到 8×8=64 个 token，每个 token 的维度是 128。
- 序列最前面再拼一个可学习的 class token。位置编码和这 65 个 token 相加，让模型知道谁在前、谁在后。
- 每一层 Encoder 是：LayerNorm、多头注意力、残差相加、LayerNorm、MLP、残差相加。
- 注意力是 token 两两之间算相似度再加权求和。本课直接用 `nn.MultiheadAttention`，先不手写 QKV。
- 分类时只取 class token，也就是序列的第 0 个位置，送到 `Linear`。
- 卷积自带「只看局部、权重在图上共享」。ViT 没有这个偏置，小数据集上常常不如 ResNet。这是本课要观察到的现象。
- Swin 的阅读点：全局注意力在大特征图上很贵。Swin 只在窗口内做注意力，下一层把窗口错开，并用 patch merging 把分辨率减半，从而接回检测和分割。本课不实现 Swin。

## 结构与形状表

输入 `(N, 3, 32, 32)`。

| 步骤 | 输出 |
| --- | --- |
| Conv patch，k=4，s=4 | `(N, 128, 8, 8)` |
| 拉成序列 | `(N, 64, 128)` |
| 拼上 class token | `(N, 65, 128)` |
| 加上位置编码 | `(N, 65, 128)` |
| 6 个 EncoderBlock | `(N, 65, 128)` |
| 取 `x[:, 0]` 再 Linear | `(N, 10)` |

`trace_shapes` 只能看到写成 `nn.Module` 的层。class token 的拼接写在 `ViTTiny.forward` 里，不会单独占一行叶子模块日志。最终 `output` 仍应是 `(2, 10)`。

## 对应例程导读

[code/lesson13_vit.py](../code/lesson13_vit.py)。

`PatchEmbed.forward` 里 `flatten(2).transpose(1, 2)` 把 `(N, C, H, W)` 变成 `(N, H*W, C)`，这是 Transformer 要的序列格式。

`EncoderBlock` 的两次 `x + ...` 是和 ResNet 同一思想的残差，只是加在 token 序列上，不是加在特征图上。

`self.cls` 和 `self.pos` 是 `nn.Parameter`，会计入参数量，但不是 `nn.Module`，形状日志里没有它们的名字。

## 动手实验

```bash
python code/lesson13_vit.py
python code/lesson13_vit.py --mode train --epochs 30
```

和 ResNet-18 使用相同的 epoch 数。把两者的验证准确率一起写入 [results.md](../results.md)。若 ViT 更低，在备注里写：结构更新之后，小数据上不一定立刻更准。

## 思考题

1. 32×32、patch=4，token 个数为什么是 64，再加 class token 后为什么是 65？
2. 为什么分类头只用 `x[:, 0]`，而不是把 65 个 token 平均？
3. 卷积的局部连接，在这份 ViT 里被哪一个层替代了？Patch 内部呢？
