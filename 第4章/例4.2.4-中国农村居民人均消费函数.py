# Author: 史浩 浙江金融职业学院
# --------------------------------------------------------
# coding=utf-8
import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_excel(r'..\data\例题数据.xlsx', sheet_name="4.2.1")
df = df.iloc[1:, :4]  # 注意：第0行是表头，要从第1行开始获得数据
df = df.dropna()
# print(df)
#              0      1     2    3
df.columns = ['area', 'Y', 'X1', 'X2']
df.reset_index(drop=True, inplace=True)  # 把索引重新排一下
# print(df)

# 定义变量
# 用.astype(float)将EXCEL文件中的字符型转浮点型
Y = df.iloc[:, 1].astype(float).apply(np.log)
X1 = df.iloc[:, 2].astype(float).apply(np.log)
X2 = df.iloc[:, 3].astype(float).apply(np.log)

X = np.column_stack((X1, X2))
##########################################################################
# 进行多元线性回归，可以替换Y和X
#          OLS：  Y          = beta0 + beta1*X1 + beta2*X2
##########################################################################
model = sm.OLS(Y, sm.add_constant(X))  # 用add_constant加入常数项
fit = model.fit()
print(fit.summary())

e2 = fit.resid ** 2
# 异方差性检验图
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TKAgg')
plt.scatter(X2, e2)
plt.xlabel('LOG(X2)')
plt.ylabel('E^2')
plt.show()
# 异方差检验
print("=" * 50)
print("BP检验 F统计量显示系数不全为零")
print("=" * 50)
model = sm.OLS(e2, sm.add_constant(X2))  # 用add_constant加入常数项
print(model.fit().summary())
print("=" * 50)
print("White检验 F统计量显示系数不全为零")
print("=" * 50)
X = np.column_stack((X1, X2, X1 ** 2, X2 ** 2))
model = sm.OLS(e2, sm.add_constant(X))  # 用add_constant加入常数项
print(model.fit().summary())

# import statsmodels.stats.api as sms
# test = sms.het_breuschpagan(fit.resid, fit.model.exog)  # result是训练好的模型
# print(test)
