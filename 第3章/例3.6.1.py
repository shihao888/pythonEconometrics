# Author: 史浩 浙江金融职业学院
# --------------------------------------------------------
import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_excel(r'..\data\例题数据.xlsx', sheet_name="3.6.1")
df = df.iloc[:, :9]
df = df.dropna()
# print(df)
#              0      1    2     3     4     5    6     7     8
df.columns = ['area', 'Y', 'X1', 'X2', 'X3', 'Y', 'X1', 'X2', 'X3']
df.reset_index(drop=True, inplace=True)  # 把索引重新排一下
# print(df)

# 定义变量
# 用.astype(float)将EXCEL文件中的字符型转浮点型
D1 = pd.DataFrame(data={"city": [1 for i in range(1, df.shape[0])]})  # 农村居民=1
D0 = pd.DataFrame(data={"city": [0 for i in range(1, df.shape[0])]})  # 城市居民=0
D = np.row_stack((D1, D0))

Y = np.row_stack((df.iloc[:, 1].astype(float), df.iloc[:, 5].astype(float)))
print(Y)
X1 = np.row_stack((df.iloc[:, 2].astype(float), df.iloc[:, 6].astype(float)))
X2 = np.row_stack((df.iloc[:, 3].astype(float), df.iloc[:, 7].astype(float)))
X3 = np.row_stack((df.iloc[:, 4].astype(float), df.iloc[:, 8].astype(float)))
X1X2X3 = np.column_stack((D, X1, np.dot(D, X1), X2, np.dot(D, X2), X3, np.dot(D, X3)))
##########################################################################
# 进行多元线性回归，可以替换Y和X
#          OLS：  Y          = beta0 + beta1*X1 + beta2*X2
##########################################################################
model = sm.OLS(Y, sm.add_constant(X1X2X3))  # 用add_constant加入常数项
print(model.fit().summary(xname=['const', 'D', 'X1', 'D*X1', 'X2', 'D*X2', 'X3', 'D*X3']))  # 用自己的名称命名常数和各个解释变量
