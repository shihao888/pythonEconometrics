# Author: 史浩 浙江金融职业学院
# --------------------------------------------------------
import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_excel(r'..\data\例题数据.xlsx', sheet_name="4.1.2")
df = df.iloc[:, :8]
df = df.dropna()
# print(df)
#              0      1     2    3     4     5      6    7
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


Z = np.column_stack((X1, X2, X3, X4, X5, X6))
print(np.corrcoef((X1, X2, X3, X4, X5, X6)))
##########################################################################
# 进行多元线性回归，可以替换Y和X
#          OLS：  Y          = beta0 + beta1*X1 + beta2*X2
##########################################################################
model = sm.OLS(Y, sm.add_constant(Z))  # 用add_constant加入常数项
print(model.fit().summary(xname=['const', 'lnX1', 'lnX2', 'lnX3', 'lnX4', 'lnX5', 'lnX6']))  # 用自己的名称命名常数和各个解释变量

# 对残差的正态性检验（雅克-贝拉检验Jarque-Bera test）:H0:残差为正态分布，H1：残差不是正态分布
