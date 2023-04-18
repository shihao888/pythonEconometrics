# Author: 史浩 浙江金融职业学院
# --------------------------------------------------------
import pandas as pd
import numpy as np
import statsmodels.api as sm

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
Y = Y/L
K = K/L
Y = Y.apply(np.log)
K = K.apply(np.log)

##########################################################################
# 进行多元线性回归，可以替换Y和X
#          OLS：  Y          = beta0 + beta1*X1 + beta2*X2
##########################################################################
model = sm.OLS(Y, sm.add_constant(K))  # 用add_constant加入常数项
print(model.fit().summary(xname=['const', 'ln(K/L)']))  # 用自己的名称命名常数和各个解释变量
