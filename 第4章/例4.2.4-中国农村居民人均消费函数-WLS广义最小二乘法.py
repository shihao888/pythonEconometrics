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
model = sm.OLS(Y, sm.add_constant(X))  # 用add_constant加入常数项
fit = model.fit()
print(fit.summary())

e2 = fit.resid ** 2
print("=" * 50)
print("加权最小二乘法:求权重")
print("=" * 50)

lne2 = e2.apply(np.log)
X = np.column_stack((X2, X2 ** 2))
model = sm.OLS(lne2, sm.add_constant(X))  # 用add_constant加入常数项
fit = model.fit()
print(fit.summary())
y_hat = fit.predict(sm.add_constant(X))
w = 1 / (np.exp(y_hat))  # 1/residual's square
print(y_hat)
print("=" * 50)
print("权重:")
print("=" * 50)
print(w)
print("=" * 50)
print("加权最小二乘法")
print("=" * 50)
# OLS是minimize sum(residuals^2),
# 而WLS是minimize sum(w*residuals^2),即将权数与残差平方相乘后再求和，所以要先定义权重。
X = np.column_stack((X1, X2))
mod_wls = sm.WLS(Y, sm.add_constant(X), weights=w)
res_wls = mod_wls.fit()
print(res_wls.summary())

# 检验是否已经不存在异方差性
print(res_wls.resid)
E2 = res_wls.resid ** 2 * w  # WLS计算出来的残差也需要乘以w才是真正的残差
w = np.sqrt(w)  # 这里才开根号
X = np.column_stack((w, w*X1, w*X2))
mod = sm.OLS(E2, sm.add_constant(X))  # There is no const, cause const is replaced by w*const
print(mod.fit().summary())
