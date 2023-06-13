# Author: 史浩 浙江金融职业学院
# --------------------------------------------------------
# 导入需要用的数据包，没有下载依赖包的，可以到Anaconda命令窗口，或者cmd中通过pip install进行下载安装
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Y=5+2⋅X+3⋅X^2

nsample = 50  # 定义样本量
x = np.linspace(0, 10, nsample)  # 生成以0为开头，以10为结尾的nsample个等差数据
X = np.column_stack((x, x ** 2))  # 按照需求去添加一列x平方，这样就有x和x平方这2列数据了
X = sm.add_constant(X)  # 再添加一个全为1的列

beta = np.array([5, 2, 3])  # 定义回归系数咯
e = np.random.normal(size=nsample)  # 来生成size为nsample个的高斯噪声
y = np.dot(X, beta) + e  # 通过计算矩阵的点乘内积，再加上高斯噪声的方式，生成需要用的模拟数据
model = sm.OLS(y, X)  # 利用模拟数据的因变量y与自变量们(1,x,x^2)进行最小二乘回归，返回model
results = model.fit()  # 拟合数据，返回拟合结果啦
print(results.params)  # 查看拟合出的回归系数
print(results.summary())  # 查看拟合结果分析)

# 绘制图形
y_fitted = results.fittedvalues  # 给出y的拟合值
fig, ax = plt.subplots(figsize=(8, 6))  # 绘制画布
ax.plot(x, y, 'o', label='data')  # 绘制原始数据的图
ax.plot(x, y_fitted, 'r--.', label='OLS')  # 绘制拟合数据图
ax.legend(loc='best')  # 生成图例
plt.show()
