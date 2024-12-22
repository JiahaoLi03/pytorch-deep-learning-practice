import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 用于绘制 3D 图形，Axes3D 是 Matplotlib 中绘制 3D 图的工具

# 这里设函数为 y = 6 * x + 2
x_data = [1.0, 2.0, 3.0]
y_data = [8.0, 14.0, 20.0]


def forward(x):
    return x * w + b


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


w_list = []
b_list = []
mse_list = []

# 穷举法 使用两个 for 循环遍历所有可能的 w 和 b 的值
for w in np.arange(0.0, 8.1, 0.1):
    for b in np.arange(0.0, 8.1, 0.1):
        print("w = ", w, "b = ", b)
        l_sum = 0  # 初始化每次权重和偏置组合下的总损失
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(x_val)
            loss_val = loss(x_val, y_val)
            l_sum += loss_val
            print('\t', x_val, y_val, y_pred_val, loss_val)
        print("MSE = ", l_sum / 3)

        w_list.append(w)
        b_list.append(b)
        mse_list.append(l_sum / 3)

# 创建一个图形 (fig) 和一个 3D 子图 (ax)   projection='3d' 表示这是一个 3D 图
fig = plt.figure()
# 将一个子图 (Axes) 添加到 fig 中
# add_subplot 的参数可以指定子图的位置和排列方式
# 111：是一个特殊的参数，表示在 1行 1列的网格中创建第一个子图
ax = fig.add_subplot(111, projection='3d')

# 绘制线框图 w_list --> x轴  b_list --> y轴  mse_list --> z轴
ax.plot(w_list, b_list, mse_list, 'o-', color='red')
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('Loss')
plt.show()










