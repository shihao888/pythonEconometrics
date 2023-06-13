# Author: 史浩 浙江金融职业学院
# --------------------------------------------------------
# 导入要用的包，没有下载包的要用pip install安装对应的包
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

nsample = 20  # 定义样本量
x = np.linspace(0, 10, nsample)  # 生成0-10之间的20个数，作为样本
X = sm.add_constant(x)  # 增加一列，默认值为1
# print(x)

# β0,β1分别设置成2,5
beta = np.array([2, 5])  # beta 是2*1的
# print(beta)

# 生成误差项
e = np.random.normal(size=nsample)  # 生成一组误差项，生成高斯噪声数据20个
# print(e)

# 利用实际值y
y = np.dot(X, beta) + e  # dot是用来算矩阵内积的，20*2 2*1 -> 20*1，接着添加高斯噪声，产生模拟用的数据
# 调用statsmodels中的方法OLS实现最小二乘法
model = sm.OLS(y, X)  # sm.OLS最小二乘，返回一个model
# 拟合数据
res = model.fit()  # fit就是拟合数据
# 回归系数
print(res.params)  # params查看拟合出的回归系数，对比设计的系数2,5，还是有一定的误差，这与数据的随机性有关
# 全部结果
print(res.summary())  # summary()返回全部结果

# 绘制图形
# 拟合的估计值
y_ = res.fittedvalues
fig, ax = plt.subplots(figsize=(8, 6))  # 设置画布
ax.plot(x, y, 'o', label='data')  # 原始数据
ax.plot(x, y_, 'r--.', label='test')  # 拟合数据
ax.legend(loc='best')  # 展示图例
plt.show()
