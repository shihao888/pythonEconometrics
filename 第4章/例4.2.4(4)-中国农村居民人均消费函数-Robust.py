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
                                 # HC1 --- white robust
fit = model.fit(cov_type='HC1')  # ‘HC0’, ‘HC1’, ‘HC2’, ‘HC3’: heteroscedasticity robust covariance
                                 # ‘HAC’: heteroskedasticity-autocorrelation robust covariance
print(fit.summary())
