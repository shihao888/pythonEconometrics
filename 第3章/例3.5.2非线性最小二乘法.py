# Author: 史浩 浙江金融职业学院
# --------------------------------------------------------
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import least_squares  # 引入最小二乘法算法

# from scipy.optimize import leastsq  # 这个已经过时不用了

df = pd.read_excel(r'..\data\例题数据.xlsx', sheet_name="3.5.1")
df = df.iloc[:, :5]
df = df.dropna()
# print(df)
#              0    1             2    3    4
df.columns = ['id', 'industrial', 'Y', 'K', 'L']
df.reset_index(drop=True, inplace=True)  # 把索引重新排一下
# print(df)
# 定义变量
# 用.astype(float)将EXCEL文件中的字符型转浮点型
Y = df.iloc[:, 2].astype(float)
K = df.iloc[:, 3].astype(float)
L = df.iloc[:, 4].astype(float)
Ydata = np.array([Y]).astype(np.float64)
xdata = np.array([K, L]).astype(np.float64)
init_param = np.array([2, 0.6, 0.3]).astype(np.float64)


###################################################################
# 需要拟合的函数func及残差e
def func(param, x_data):
    nextParam = np.ones(3).astype(np.float64)  # 初始化
    # 初始化之后，填充数据
    for j in range(3):
        nextParam[j] = np.array(param[j]).astype(np.float64)

    # 设置自变量X1 X2 X3 ......
    X1 = np.array(x_data[0]).astype(np.float64)
    X2 = np.array(x_data[1]).astype(np.float64)

    # 如果方程是这样 Y= next_param0+next_param1*X1+next_param2*X2^next_param3+next_param4*ln(X3)
    # 返回就这样    return next_param[0] +  next_param[1] * X1 + next_param[2] * (X2 ** next_param[3]) + next_param[4] * np.log(X3)
    # log表示ln以e为底, 而log10表示以10为底

    f = np.exp(nextParam[0]) * (X1 ** nextParam[1]) * (X2 ** nextParam[2])
    f1 = np.array(f)
    # print(f'f1={f1}')
    return f1


# 最小二乘法中的残差 = e = 预测值yhat(利用估计的参数p和已知x计算得到) - 实际值y
def e(p, x, y):  # p:param参数
    r = np.array(func(p, x)).astype(np.float64) - np.array(y).astype(np.float64)
    r1 = np.array(r).flatten()
    # print(f'r={r1}')
    return r1


s = least_squares(e, init_param, args=(xdata, Ydata), verbose=2)
print(s.x)
opt_param = s.x  # 得到了拟合参数

# 将得到的拟合参数代入计算
e = []
Y_Fit = []
for i in range(0, len(Ydata)):
    Y_Fit.append(np.exp(opt_param[0]) * (xdata[0] ** opt_param[1]) * (xdata[1] ** opt_param[2]))
    e.append(Y_Fit[i] - Ydata[i])
# 列表转换为矩阵
Y_Fit = np.array(Y_Fit).flatten()
e = np.array(e).flatten()

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = 'False'
titlex = ['1#', '2#', '3#', '4#', '6#', '7#', '8#', '3#和8#']
colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black', 'cyan']


def Fitting_drawing(flag, Y_Actual, Y_Virtual, Err):  # i,Y,Y_Fit,e

    M1 = max(Y_Actual)
    print(M1)
    M2 = max(Y_Virtual)
    print(M2)
    Y1_MAX = max(M1, M2)

    M1 = min(Y_Actual)
    M2 = min(Y_Virtual)
    Y1_MIN = min(M1, M2)

    plt.figure()
    plt.subplot(211)
    # 设置范围
    plt.xlim(1, len(Y_Virtual))
    plt.ylim(Y1_MIN - 10, Y1_MAX + 10)
    plt.grid(alpha=0.2, linestyle=':')
    # 设置标题标签
    plt.title('生产函数')
    plt.xlabel(titlex[flag - 2] + '非线性拟合')

    # 横坐标
    X = np.arange(1, len(Y_Virtual) + 1)
    plt.plot(X, Y_Virtual, label='拟合值', color='red', linestyle='-', marker='o', markersize=5, alpha=1)  # 对线添加标图例
    plt.plot(X, Y_Actual, label='实际值', color='blue', linestyle='-', marker='o', markersize=5, alpha=1)  # 对线添加标图例

    # X = L
    # plt.scatter(X, Y_Virtual, label='拟合值', color='red', linestyle='-', marker='o', alpha=0.5)  # 对线添加标图例
    # plt.scatter(X, Y_Actual, label='实际值', color='blue', linestyle='-', marker='o', alpha=1)  # 对线添加标图例

    plt.legend()  # 生成图例
    plt.subplot(212)
    # 设置范围
    plt.xlim(1, len(Err))
    plt.ylim(min(Err) - 10, max(Err) + 10)
    plt.grid(alpha=0.2, linestyle=':')
    # 设置标题标签
    plt.title('S -- 误差')
    plt.plot(X, Err, label='误差', color='green', linestyle='-', marker='o', markersize=5, alpha=1)  # 对线添加标图例
    plt.legend()  # 生成图例
    plt.subplots_adjust(wspace=0.3, hspace=0.5)  # 每个图之间的间距（高，宽）
    plt.show()


# 绘制
for i in range(1):
    Fitting_drawing(i, Y, Y_Fit, e)
