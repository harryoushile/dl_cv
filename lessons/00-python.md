# 第 0 课：Python 地基

## 这一代要解决的问题

后面每个网络都是一个 Python 类。还不会写函数和类，就读不懂 `nn.Module`。这一课只学后面每一份代码都会碰到的语法。

## 知识要点

- 变量、列表、字典、`for`、`if`、函数。
- 类：`__init__` 保存数据，方法的第一个参数是 `self`。
- 读报错时先看最后一行的错误类型，再往上找自己文件的行号。
- 列表用来按顺序保存多次成绩；字典用来按名字取值。

足够支撑本课的练习是一个 `Student` 类，不需要先学完一整本 Python。

## 结构与形状表

这一课没有张量。把「对象里有什么」写成表：

| 名字 | 存在哪 | 作用 |
| --- | --- | --- |
| `name` | `self.name` | 学生姓名 |
| `scores` | `self.scores` | 三次成绩的列表 |
| `average` | 方法 | 返回平均分 |

## 对应例程导读

本课没有 `code/` 文件。在交互环境或一个临时 `student.py` 里写：

```python
class Student:
    def __init__(self, name, scores):
        self.name = name
        self.scores = scores

    def average(self):
        return sum(self.scores) / len(self.scores)

s = Student("lin", [80, 90, 100])
print(s.name, s.average())
```

对照后面的网络：`__init__` 里创建层，`forward` 相当于这里的 `average`，是对象被调用时真正做的计算。PyTorch 的 `nn.Module` 用 `forward` 这个固定名字，所以训练代码不用知道网络内部有几层。

## 动手实验

1. 给 `Student` 增加方法 `passed`，平均分不低于 60 时返回 `True`。
2. 故意把 `self.scores` 写成 `scores`，运行并指出报错行。
3. 写一个函数 `count_above(scores, threshold)`，返回不低于门槛的成绩个数。

## 思考题

1. 为什么方法里要用 `self.scores`，而不是直接写 `scores`？
2. `Student("lin", [])` 调用 `average` 会发生什么？
3. 如果有 30 个学生，用列表保存 `Student` 对象和用字典按姓名保存，取某一个人的平均分时有什么差别？
