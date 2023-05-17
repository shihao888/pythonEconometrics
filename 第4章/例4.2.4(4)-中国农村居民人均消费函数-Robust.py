# Author: 史浩 浙江金融职业学院
# import sys
# from PySide6 import QtWidgets, QtCore, QtGui
# from PySide6.QtWidgets import QWidget, QListWidget, QListWidgetItem, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
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
model1 = sm.OLS(Y, sm.add_constant(X))  # 用add_constant加入常数项
# HC1 --- white robust
fit_non_robust = model1.fit(use_t=True)  # ‘HC0’, ‘HC1’, ‘HC2’, ‘HC3’: heteroscedasticity robust covariance
# ‘HAC’: heteroskedasticity-autocorrelation robust covariance
print(fit_non_robust.summary())

##########################################################################
# 进行多元线性回归，可以替换Y和X
#          OLS：  Y          = beta0 + beta1*X1 + beta2*X2
##########################################################################
model2 = sm.OLS(Y, sm.add_constant(X))  # 用add_constant加入常数项
# HC0 --- white robust; use_t --- t statistic used
fit_hc0 = model2.fit(cov_type='HC0', use_t=True)  # ‘HC0’, ‘HC1’, ‘HC2’, ‘HC3’: heteroscedasticity robust covariance
# ‘HAC’: heteroskedasticity-autocorrelation robust covariance
print(fit_hc0.summary())

# 比较标准误
print(fit_non_robust.bse)
print(fit_hc0.bse)
print("稳健标误一定大于传统标误。如果是小于，那么很大概率是有限样本（finite samplt）偏误。结论不可靠。")
print("推荐阅读https://www.mostlyharmlesseconometrics.com/2010/12/heteroskedasticity-and-standard-errors-big-and-small/")
# 用小写字母ols回归，就是采用formula形式的
# import statsmodels.formula.api as smf
# lm = smf.ols('Y~X1+X2', data=df).fit()
# print(lm.summary().tables[1])

# https://blog.csdn.net/weixin_39858881/article/details/89488371
# HC0_se – White’s (1980) heteroskedasticity robust standard errors. Defined as sqrt(diag(X.T X)^(-1)X.T diag(e_i^(2)) X(X.T X)^(-1) where e_i = resid[i] HC0_se is a cached property. When HC0_se or cov_HC0 is called the RegressionResults instance will then have another attribute het_scale, which is in this case is just resid**2.
# HC1_se – MacKinnon and White’s (1985) alternative heteroskedasticity robust standard errors. Defined as sqrt(diag(n/(n-p)*HC_0) HC1_see is a cached property. When HC1_se or cov_HC1 is called the RegressionResults instance will then have another attribute het_scale, which is in this case is n/(n-p)*resid**2.
# HC2_se – MacKinnon and White’s (1985) alternative heteroskedasticity robust standard errors. Defined as (X.T X)^(-1)X.T diag(e_i^(2)/(1-h_ii)) X(X.T X)^(-1) where h_ii = x_i(X.T X)^(-1)x_i.T HC2_see is a cached property. When HC2_se or cov_HC2 is called the RegressionResults instance will then have another attribute het_scale, which is in this case is resid^(2)/(1-h_ii).
# HC3_se – MacKinnon and White’s (1985) alternative heteroskedasticity robust standard errors. Defined as (X.T X)^(-1)X.T diag(e_i(2)/(1-h_ii)(2)) X(X.T X)^(-1) where h_ii = x_i(X.T X)^(-1)x_i.T HC3_see is a cached property. When HC3_se or cov_HC3 is called the RegressionResults instance will then have another attribute het_scale, which is in this case is resid(2)/(1-h_ii)(2).
#     White标准误（异方差稳健的标准误）：
#     HC0:White（1980）提出的异方差稳健的标准误
#     HC1:Mackinon and White（1985）提出的异方差稳健的标准误
#     HC2:MacKinnon and White（1985）提出的异方差稳健的标准误
#     HC3:MacKinnon and White（1985）提出的异方差稳健的标准误
#     HAC:Newey-West标准误（异方差自相关稳健的标准误）：
# results = OLS(...).fit(cov_type='cluster', cov_kwds={'groups': mygroups}
#
