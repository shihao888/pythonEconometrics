# coding=utf-8
import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_excel(r'..\data\例题数据.xlsx', sheet_name="4.3.1")
df = df.iloc[1:, :6]  # 注意：第0行是表头，要从第1行开始获得数据
df = df.dropna()
# print(df)
#              0      1     2    3
df.columns = ['area', 'Q', 'Y', 'P', 'tax', 'taxes']
df.reset_index(drop=True, inplace=True)  # 把索引重新排一下
# print(df)

# 定义变量
# 用.astype(float)将EXCEL文件中的字符型转浮点型
Q = df.iloc[:, 1].astype(float).apply(np.log)
Y = df.iloc[:, 2].astype(float).apply(np.log)
P = df.iloc[:, 3].astype(float).apply(np.log)
tax = df.iloc[:, 4].astype(float)
taxes = df.iloc[:, 5].astype(float)
Z = np.column_stack((Y, tax))  # 注意这里把Y也放入
##########################################################################
# 进行多元线性回归，可以替换Y和X
#          OLS：  Y          = beta0 + beta1*X1 + beta2*X2
##########################################################################
model1 = sm.OLS(P, sm.add_constant(Z))  # 用add_constant加入常数项
fit1 = model1.fit()
# print(fit1.summary())
p_hat = fit1.predict(sm.add_constant(Z))  # predict与这个p_hat = fit1.fittedvalues同等效果

X = np.column_stack((Y, p_hat))  # np.column_stack需要至少2个参数
model2 = sm.OLS(Q, sm.add_constant(X))  # 用add_constant加入常数项
fit2 = model2.fit()
print(fit2.summary())

Z = np.column_stack((Y, tax, taxes))  # 注意这里把Y也放入
##########################################################################
# 进行多元线性回归，可以替换Y和X
#          OLS：  Y          = beta0 + beta1*X1 + beta2*X2
##########################################################################
model2 = sm.OLS(P, sm.add_constant(Z))  # 用add_constant加入常数项
fit2 = model2.fit()
# print(fit2.summary())
p_hat = fit2.predict(sm.add_constant(Z))  # predict与这个p_hat = fit2.fittedvalues同等效果

X = np.column_stack((Y, p_hat))  # np.column_stack需要至少2个参数
model2 = sm.OLS(Q, sm.add_constant(X))  # 用add_constant加入常数项
fit2 = model2.fit()
print(fit2.summary())
