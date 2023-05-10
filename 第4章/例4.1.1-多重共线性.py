# Author: 史浩 浙江金融职业学院
# --------------------------------------------------------
import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_excel(r'..\data\例题数据.xlsx', sheet_name="4.1.2")
df = df.iloc[:, :8]
df = df.dropna()
# print(df)
#              0      1     2    3     4     5     6     7
df.columns = ['area', 'Y', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6']
df.reset_index(drop=True, inplace=True)  # 把索引重新排一下
# print(df)

# 定义变量
# 用.astype(float)将EXCEL文件中的字符型转浮点型
Y = df.iloc[:, 1].astype(float).apply(np.log)
X1 = df.iloc[:, 2].astype(float).apply(np.log)
X2 = df.iloc[:, 3].astype(float).apply(np.log)
X3 = df.iloc[:, 4].astype(float).apply(np.log)
X4 = df.iloc[:, 5].astype(float).apply(np.log)
X5 = df.iloc[:, 6].astype(float).apply(np.log)
X6 = df.iloc[:, 7].astype(float).apply(np.log)
X = np.column_stack((X1, X2, X3, X4, X5, X6))
##########################################################################
# 进行多元线性回归，可以替换Y和X
#          OLS：  Y          = beta0 + beta1*X1 + beta2*X2
##########################################################################
model = sm.OLS(Y, sm.add_constant(X))  # 用add_constant加入常数项
print(model.fit().summary())
print("R2太高，而且F值太大，说明有多重共线性。")
print("检验其相关系数：")
# 将多个列合并成一个表，也就是将多个Series对象合并成一个DataFrame对象。
# 主要推荐的函数有pd.concat()
# 和 pd.DataFrame(list(zip(s1, s2, s3)))
X_df = pd.concat([X1, X2, X3, X4, X5, X6], axis=1, ignore_index=True)  # 列合并
# 相关系数：X_df.corr返回结果是一个数据框corr_df，存放的是相关系数矩阵corr_df
corr_df = X_df.corr()
print('方法一相关系数矩阵：')
print(corr_df)
# 另外一种方法是
print('方法二相关系数矩阵：')
print(np.corrcoef((X1, X2, X3, X4, X5, X6)))

# 可利用逐步回归筛选并剔除引起多重共线性的变量，
# 其具体步骤如下：
# 先用被解释变量对每一个所考虑的解释变量做简单回归，
# 然后以对被解释变量贡献最大的解释变量所对应的回归方程为基础，
# 再逐步引入其余解释变量。经过逐步回归，
# 使得最后保留在模型中的解释变量既是重要的，又没有严重多重共线性。
# 导入要用到的各种包和函数

import numpy as np

import pandas as pd

from sklearn import datasets, linear_model

from math import sqrt

import matplotlib.pyplot as plt

