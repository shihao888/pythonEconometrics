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
D = pd.concat([D1, D0]).reset_index(drop=True)  # drop=True 排序后去掉原来索引
# D1 D0 D都为DataFrame
# 最后一行是平均值所以舍去不用，因此df.iloc[:-1,中用-1
# df.iloc[:-1, 1].astype(float) 产生Series
Y = pd.concat([df.iloc[:-1, 1].astype(float), df.iloc[:-1, 5].astype(float)]).reset_index(drop=True)
X1 = pd.concat([df.iloc[:-1, 2].astype(float), df.iloc[:-1, 6].astype(float)]).reset_index(drop=True)
X2 = pd.concat([df.iloc[:-1, 3].astype(float), df.iloc[:-1, 7].astype(float)]).reset_index(drop=True)
X3 = pd.concat([df.iloc[:-1, 4].astype(float), df.iloc[:-1, 8].astype(float)]).reset_index(drop=True)
DX1 = D.mul(X1, axis=0)  # DataFrame 乘以 Series
DX2 = D.mul(X2, axis=0)
DX3 = D.mul(X3, axis=0)
# print(D.mul(X1, axis=0))
X = np.column_stack((D, X1, DX1, X2, DX2, X3, DX3))
##########################################################################
# 进行多元线性回归，可以替换Y和X
#          OLS：  Y          = beta0 + beta1*X1 + beta2*X2
##########################################################################
model = sm.OLS(Y, sm.add_constant(X))  # 用add_constant加入常数项
print(model.fit().summary(xname=['const', 'D', 'X1', 'D*X1', 'X2', 'D*X2', 'X3', 'D*X3']))  # 用自己的名称命名常数和各个解释变量
