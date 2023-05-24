# Author: 史浩 浙江金融职业学院
# --------------------------------------------------------
# coding=utf-8
import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_excel(r'..\data\例题数据.xlsx', sheet_name="4.3.1")
df = df.iloc[1:, :6]  # 注意：第0行是表头，要从第1行开始获得数据
df = df.dropna()
# print(df)
#              0      1     2    3
df.columns = ['state', 'Q', 'Y', 'P', 'TAX', 'TAXES']
df.reset_index(drop=True, inplace=True)  # 把索引重新排一下
# print(df)

# 定义变量
# 用.astype(float)将EXCEL文件中的字符型转浮点型
Q = df.iloc[:, 1].astype(float).apply(np.log)
Y = df.iloc[:, 2].astype(float).apply(np.log)
P = df.iloc[:, 3].astype(float).apply(np.log)
TAX = df.iloc[:, 4].astype(float)
TAXES = df.iloc[:, 5].astype(float)

X = np.column_stack((Y, P))
##########################################################################
# 进行多元线性回归，可以替换Y和X
#          OLS：  Y          = beta0 + beta1*X1 + beta2*X2
##########################################################################
model = sm.OLS(Q, sm.add_constant(X))  # 用add_constant加入常数项
# 异方差稳健的标准误 ：
#     HC0:White（1980）提出的异方差稳健的标准误
#     HC1:Mackinon and White（1985）提出的异方差稳健的标准误
#     HC2:MacKinnon and White（1985）提出的异方差稳健的标准误
#     HC3:MacKinnon and White（1985）提出的异方差稳健的标准误
#     HAC:Newey-West标准误（异方差自相关稳健的标准误）
fit = model.fit(cov_type='HC0', use_t=True)
# print(fit.summary())
Q_hat = fit.predict(sm.add_constant(X))
Q2 = Q_hat ** 2
X = np.column_stack((Y, P, Q2))
##########################################################################
# 进行多元线性回归，可以替换Y和X
#          OLS：  Y          = beta0 + beta1*X1 + beta2*X2
##########################################################################
model = sm.OLS(Q, sm.add_constant(X))  # 用add_constant加入常数项
# fit = model.fit(cov_type='HC0', use_t=True)
# print(fit.summary())


Q = df.iloc[:, 1].astype(float)
Y = df.iloc[:, 2].astype(float)
P = df.iloc[:, 3].astype(float)
X = np.column_stack((Y, P))
##########################################################################
# 进行多元线性回归，可以替换Y和X
#          OLS：  Y          = beta0 + beta1*X1 + beta2*X2
##########################################################################
model1 = sm.OLS(Q, sm.add_constant(X))  # 用add_constant加入常数项
fit = model1.fit(cov_type='HC0', use_t=True)
print(fit.summary())
Q_hat = fit.predict(sm.add_constant(X))
Q2 = Q_hat ** 2
X = np.column_stack((Y, P, Q2))
##########################################################################
# 进行多元线性回归，可以替换Y和X
#          OLS：  Y          = beta0 + beta1*X1 + beta2*X2
##########################################################################
model = sm.OLS(Q, sm.add_constant(X))  # 用add_constant加入常数项
fit = model.fit(cov_type='HC0', use_t=True)
print(fit.summary())
