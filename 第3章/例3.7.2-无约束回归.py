# Author: 史浩 浙江金融职业学院
# --------------------------------------------------------
import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_excel(r'..\data\例题数据.xlsx', sheet_name="3.7.1")
df = df.iloc[:, :10]
df = df.dropna()
# print(df)
#              0      1    2    3     4     5      6      7      8     9
df.columns = ['area', 'Q', 'P', 'P1', 'P2', 'P01', 'P02', 'P03', 'P0', 'X']
df.reset_index(drop=True, inplace=True)  # 把索引重新排一下
# print(df)

# 定义变量
# 用.astype(float)将EXCEL文件中的字符型转浮点型
Q = df.iloc[:, 1].astype(float).apply(np.log)
P = df.iloc[:, 2].astype(float)
P1 = df.iloc[:, 3].astype(float)
P2 = df.iloc[:, 4].astype(float)
P01 = df.iloc[:, 5].astype(float)
P02 = df.iloc[:, 6].astype(float)
P03 = df.iloc[:, 7].astype(float)
P0 = df.iloc[:, 8].astype(float)
X = df.iloc[:, 9].astype(float)

Z = np.column_stack(((X/P0).apply(np.log), P1/P, P2/P, P01, P02, P03))
##########################################################################
# 进行多元线性回归，可以替换Y和X
#          OLS：  Y          = beta0 + beta1*X1 + beta2*X2
##########################################################################
model = sm.OLS(Q, sm.add_constant(Z))  # 用add_constant加入常数项
print(model.fit().summary(xname=['const', 'ln(X/P0)', 'P1/P', 'P2/P', 'P01', 'P02', 'P03']))  # 用自己的名称命名常数和各个解释变量
