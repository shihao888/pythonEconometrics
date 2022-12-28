# Author: 史浩 浙江金融职业学院
# --------------------------------------------------------
# 导入需要用的数据包，没有下载依赖包的，可以到Anaconda命令窗口，或者cmd中通关pip install进行下载安装
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd

nsample = 50
groups = np.zeros(nsample, int)  # 构造一个元素数量为50的int型0向量

groups[20:40] = 1  # 将groups中20:40的数据改成1
groups[40:] = 2  # 将groups中40之后的数据改成2
dummy = pd.get_dummies(
    pd.Categorical(groups))  # 将groups设置成categorical格式，也就是按照取值，比如0就设成[1.,0.,0.]，1就设成[0.,1.,0.]，2就设成[0.,0.,1.]
# dummy = sm.categorical(groups, drop=True)  # deprecated
# dummy = pd.DataFrame(dummy).iloc[:, 1:]
# print(dummy.iloc[:, 1:])
# Y=5+2X+3Z1+6⋅Z2+9⋅Z3.

x = np.linspace(0, 20, nsample)  # 生成nsample个0-20的等差数列数据咯
X = np.column_stack((x, dummy))  # 将x与dummy合并到一起，显然x只是1列，而dummy有3列，这样子合并起来就有4列啦
X = sm.add_constant(X)  # 再在最左边添加一个全为1的列
beta = [5, 2, 3, 6, 9]  # 生成回归系数
e = np.random.normal(size=nsample)  # 生成nsample个高斯噪声
y = np.dot(X, beta) + e  # 生成模拟数据
result = sm.OLS(y, X).fit()  # OLS进行最小二乘回归，然后会返回一个model对象，调用它的fit()方法，拟合数据，返回拟合结果
print(result.params)
print(result.summary())  # 查看拟合结果

# # 绘制图形
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(x, y, 'o', label="data")
# ax.plot(x, result.fittedvalues, 'r--.', label="OLS")
# ax.legend(loc='best')  # 添加图例
# plt.show()
