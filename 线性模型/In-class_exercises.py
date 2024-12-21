import numpy as np
import matplotlib.pyplot as plt

# 数据集（x_data 和 y_data 表示输入和输出数据
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 定义模型的前向传播（即计算预测值）
def forward(x):
    return x * w  # 线性模型：y = w * x   w --> 待求参数


def loss(x, y):
    y_pred = forward(x)  # 通过前向传播计算预测值 y_pred
    return (y_pred - y) ** 2  # 计算预测值与真实值的平方误差（真实值与预测值的差的平方）


# 穷举法 ----> 尝试不同的 w 值，寻找使损失最小的 w
w_list = []  # 存储不同的 w 值
mse_list = []  # 存储每个 w 对应的平均损失(MSE)

# 遍历 w 从 0.0 -> 4.0，每次增加 0.1
for w in np.arange(0.0, 4.1, 0.1):  # np.arange 生成等差数列，范围 0.0 -> 4.0，步长为 0.1
    print("w = ", w)  # 打印当前尝试的 w 值
    l_sum = 0  # 初始化损失累加器

    # 遍历数据集中的每个数据点
    for x_val, y_val in zip(x_data, y_data):  # zip 将 x_data 和 y_data 打包成元组逐个迭代
        y_pred_val = forward(x_val)  # 计算当前 w 下的预测值

        # 这里 loss函数 中会计算x_val前向传播后的预测值，所以传入的是 x_val 而不是 y_pred_val
        loss_val = loss(x_val, y_val)  # 计算损失（真实值和预测值之间的误差）
        l_sum += loss_val  # 累加损失
        print('\t', x_val, y_val, y_pred_val, loss_val)

    print("MSE = ", l_sum / 3)  # 计算平均损失（数据集中有 3 个点）
    w_list.append(w)  # 将当前 w 值添加到 w_list
    mse_list.append(l_sum / 3)  # 将当前对应的 MSE 添加到 mse_list


# 绘制损失函数曲线（w 与 MSE 之间的关系）
plt.plot(w_list, mse_list)  # 绘制折线图，横轴为 w，纵轴为 MSE
plt.ylabel('Loss')  # y 轴标签
plt.xlabel('w')  # x 轴标签
plt.show()  # 显示绘图结果
