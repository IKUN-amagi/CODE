import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def f(w, x):
    # 预测输出值
    return x.dot(w)


def mse(fx, y):
    # 均方误差函数
    m = len(fx)
    return 1/(2*m) * np.sum((fx - y)**2)


def compute_gradient(x, y, w):
    # 计算梯度，包括偏置项
    m = len(x)
    # 添加偏置项，x0为全1的列
    fx = f(w,x)
    error = fx - y  # 计算预测值与实际值之间的误差
    gradient = (1/m) * x.T.dot(error)  # 计算梯度
    return gradient


def batch_gradient_descent(x, y, w_init, alpha, epoch):
    # 批量梯度下降算法
    w = w_init
    for i in range(epoch):
        gradient = compute_gradient(x, y, w)
        w = w - alpha * gradient

    return w


if __name__ == "__main__":
    # 读取数据
    data = pd.read_csv('regress_data1.csv')
    population = data['人口'].values  # 转换为numpy数组
    revenue = data['收益'].values     # 转换为numpy数组

    # 定义各项参数，初始参数，学习率，迭代数
    w0 = np.zeros(2)  # 将偏置项b吸收
    alpha = 0.01
    epoch = 5000
    # 相应的，为偏置项的吸收处理数据集
    population = np.hstack((population.reshape(-1, 1), np.ones((len(population), 1))))
    # 执行批量梯度下降
    w_final = batch_gradient_descent(population, revenue, w0, alpha, epoch)
    # 输出最终的参数
    print(f"最终参数 w: {w_final}\n均方误差 MSE:{mse(f(w_final,population), revenue)}")

    # 绘制拟合直线
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
    plt.scatter(population[:, 0], revenue, color='blue')
    plt.plot(population[:, 0], f(w_final, population), color='red')  # 拟合的直线
    plt.title('人口与收益的关系（拟合直线）')
    plt.xlabel('人口')
    plt.ylabel('收益')
    plt.grid(True)
    plt.show()
