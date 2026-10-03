# 深度学习计算机视觉课

面向零基础的大一课程。主线是顺着视觉网络的演进，把每个阶段的关键结构用 PyTorch 写出来，并看清 tensor 的形状怎么变。

## 环境

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

需要 Python 3.10 及以上。没有 GPU 时，`--mode shapes` 仍然可以检查结构。训练 CIFAR-10 时有 GPU 会快很多。

## 怎么跑

在仓库根目录执行。默认只做前向并打印每一层形状，不下载数据集。

本仓库的例程目录也叫 `code`。在仓库根目录打开 Python 再执行 `import torch` 时，这个目录会挡住标准库里的 `code` 模块。请用下面的命令运行例程，不要在仓库根目录把本文件夹放进交互解释器的搜索路径后再导入 torch。

```bash
python code/lesson04_lenet.py
python code/lesson08_resnet.py
python code/lesson15_unet.py
```

开始训练时加上 `--mode train`。MNIST 用 Adam，CIFAR-10 用带动量的 SGD 和余弦学习率。

```bash
python code/lesson04_lenet.py --mode train --epochs 5
python code/lesson08_resnet.py --mode train --epochs 30
python code/lesson08_resnet.py --mode train --epochs 30 --no-shortcut
python code/lesson15_unet.py --mode train --steps 50
```

Windows 上 DataLoader 使用 `num_workers=0`，避免子进程重复导入主脚本。

## 课程序

| 课 | 课件 | 例程 | 数据 |
| --- | --- | --- | --- |
| 0 | [Python 地基](lessons/00-python.md) | 无 | 无 |
| 1 | [图像与张量](lessons/01-tensor.md) | 无 | 无 |
| 2 | [线性分类器](lessons/02-linear.md) | `code/lesson02_linear.py` | MNIST |
| 3 | [卷积基础](lessons/03-cnn.md) | `code/lesson03_tinycnn.py` | MNIST |
| 4 | [LeNet-5](lessons/04-lenet.md) | `code/lesson04_lenet.py` | MNIST |
| 5 | [AlexNet](lessons/05-alexnet.md) | `code/lesson05_alexnet.py` | CIFAR-10 |
| 6 | [VGG](lessons/06-vgg.md) | `code/lesson06_vgg.py` | CIFAR-10 |
| 7 | [GoogLeNet](lessons/07-inception.md) | `code/lesson07_inception.py` | CIFAR-10 |
| 8 | [ResNet](lessons/08-resnet.md) | `code/lesson08_resnet.py` | CIFAR-10 |
| 9 | [Bottleneck 与消融](lessons/09-bottleneck.md) | 同上，`--no-shortcut` | CIFAR-10 |
| 10 | [DenseNet](lessons/10-densenet.md) | `code/lesson10_densenet.py` | CIFAR-10 |
| 11 | [MobileNetV2](lessons/11-mobilenet.md) | `code/lesson11_mobilenet.py` | CIFAR-10 |
| 12 | [SENet](lessons/12-senet.md) | `code/lesson12_senet.py` | CIFAR-10 |
| 13 | [ViT](lessons/13-vit.md) | `code/lesson13_vit.py` | CIFAR-10 |
| 14 | [ConvNeXt](lessons/14-convnext.md) | `code/lesson14_convnext.py` | CIFAR-10 |
| 15 | [U-Net](lessons/15-unet.md) | `code/lesson15_unet.py` | 合成圆 |
| 16 | [小项目](lessons/16-project.md) | 改已有网络 | CIFAR-10 |

训练结果填到 [results.md](results.md)。

## 读网络的五步

1. 输入尺寸是多少。
2. 下采样发生在哪一层（stride 或池化）。
3. 通道怎么变。
4. 跳连或拼接在哪，相加和拼接差在哪。
5. 分类头或分割头是什么。

`trace_shapes` 的打印必须和课件里的形状表一致，再开始训练。

## 写法约定

- 子模块在 `__init__` 里赋给 `self`，优化器才能看到它们。
- `forward` 只做计算。
- 卷积后面接 BatchNorm 时，卷积使用 `bias=False`。
- 残差相加前，两个张量的四个维度都要相同。
- `torch.cat(..., dim=1)` 拼接的是通道，其余维度必须相同。
- 训练前模型处于 `train()`，测试前调用 `eval()`，Dropout 和 BatchNorm 才会切换行为。
# dl_cv
