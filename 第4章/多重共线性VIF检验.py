# Author: 史浩 浙江金融职业学院
# --------------------------------------------------------
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.outliers_influence

df = pd.read_excel(r'..\data\例题数据.xlsx', sheet_name="4.1.2")
df = df.iloc[:, :8]
df = df.dropna()

df.columns = ['area', 'Y', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6']
df.reset_index(drop=True, inplace=True)  # 把索引重新排一下
Y = df.iloc[:, 1].astype(float).apply(np.log)
X1 = df.iloc[:, 2].astype(float).apply(np.log)
X2 = df.iloc[:, 3].astype(float).apply(np.log)
X3 = df.iloc[:, 4].astype(float).apply(np.log)
X4 = df.iloc[:, 5].astype(float).apply(np.log)
X5 = df.iloc[:, 6].astype(float).apply(np.log)
X6 = df.iloc[:, 7].astype(float).apply(np.log)
X = np.column_stack((X1, X2, X3, X4, X5, X6))
model = sm.OLS(Y, sm.add_constant(X))  # 用add_constant加入常数项
print(model.fit().summary())


# vif检验
# 一般认为VIF超过10就过大，也有严格的认为是5。
# 之前论文中找到过7.5的参考文献。具体可视情况而定，不影响上述大局即可。
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = []

X = np.column_stack((np.ones(len(X)), X))
for i in range(6):
    a = variance_inflation_factor(np.array(X), i)
    vif.insert(i, a)
print(vif)
