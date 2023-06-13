# --------------------------------------------------------
# Author: 史浩 浙江金融职业学院
# --------------------------------------------------------
# import sys
# from PySide6 import QtWidgets, QtCore, QtGui
# from PySide6.QtWidgets import QWidget, QListWidget, QListWidgetItem, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
# --------------------------------------------------------
# coding=utf-8
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy as sp

# 在使用此格式定义函数时，指定有默认值的形式参数必须在所有没默认值参数的最后，否则会产生语法错误。
# print(函数名.__defaults__) 可以打印所有默认参数
# chi2_table函数：查chi2表
def chi2_table(dof, p=0.05):
    return sp.stats.chi2.ppf(1 - p, df=dof)
     

# H0: 不存在序列相关性
def LM_TEST(Y, X):
    print('LMTEST H0: 不存在序列相关性')
    model = sm.OLS(Y, sm.add_constant(X))
    fit = model.fit()
    e_hat_t = fit.resid

    # 一阶滞后
    e_hat_t_minus_1 = e_hat_t.shift(-1)
    e_hat_t_minus_1 = e_hat_t_minus_1.iloc[:-1]  # 丢弃最后一行N/A
    e_hat_t = e_hat_t.iloc[:-1]  # 为保持数据一致，也丢弃最后一行
    X = X[:-1]  # 为保持数据一致，也丢弃最后一行
    Q = np.column_stack((X, e_hat_t_minus_1))
    model = sm.OLS(e_hat_t, sm.add_constant(Q))
    fit = model.fit()
    # 自由度为1的卡方分布
    print(fit.summary())
    print(f'fit.rsquared={fit.rsquared}')
    print(f'LM({1})={fit.rsquared/((fit.nobs - 1)  * fit.rsquared)} chi2_table_value={chi2_table(1, 0.05)}')

    # 二阶滞后
    e_hat_t_minus_2 = e_hat_t_minus_1.shift(-1)
    e_hat_t_minus_2 = e_hat_t_minus_2.iloc[:-1]  # 之前丢过一行，在之前基础上再多丢一行

    e_hat_t = e_hat_t.iloc[:-1]  # 为保持数据一致，也丢弃最后两行(之前丢过一行，在之前基础上再多丢一行)
    e_hat_t_minus_1 = e_hat_t_minus_1.iloc[:-1]
    X = X[:-1]

    R = np.column_stack((X, e_hat_t_minus_1, e_hat_t_minus_2))
    model = sm.OLS(e_hat_t, sm.add_constant(R))
    fit = model.fit()
    # 自由度为2的卡方分布
    print(f'fit.rsquared={fit.rsquared}')
    print(f'LM({2})={fit.rsquared/((fit.nobs - 2)  * fit.rsquared)} chi2_table_value={chi2_table(2, 0.05)}')

    # 三阶滞后
    e_hat_t_minus_3 = e_hat_t_minus_2.shift(-1)
    e_hat_t_minus_3 = e_hat_t_minus_3.iloc[:-1]  # 之前丢过一行，在之前基础上再多丢一行

    e_hat_t = e_hat_t.iloc[:-1]  # 为保持数据一致，也丢弃最后两行(之前丢过一行，在之前基础上再多丢一行)
    e_hat_t_minus_1 = e_hat_t_minus_1.iloc[:-1]
    e_hat_t_minus_2 = e_hat_t_minus_2.iloc[:-1]
    X = X[:-1]

    R = np.column_stack((X, e_hat_t_minus_1, e_hat_t_minus_2, e_hat_t_minus_3))
    model = sm.OLS(e_hat_t, sm.add_constant(R))
    fit = model.fit(use_t=True)
    # 自由度为3的卡方分布
    print(f'fit.rsquared={fit.rsquared}')
    print(f'LM({3})={fit.rsquared/((fit.nobs - 3)  * fit.rsquared)} chi2_table_value={chi2_table(3, 0.05)}')
