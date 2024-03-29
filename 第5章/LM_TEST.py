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
from scipy import stats
# 在使用此格式定义函数时，指定有默认值的形式参数必须在所有没默认值参数的最后，否则会产生语法错误。
# print(函数名.__defaults__) 可以打印所有默认参数
# chi2_table函数：查chi2表
def chi2_table(dof, p=0.05):
    return sp.stats.chi2.ppf(1 - p, df=dof)

# 回归方程是Y和X 最多3阶
def LM_TEST_lag3(Y, X, nocons=False):
    print('LMTEST H0: 不存在序列相关性')
    if len(Y) != len(X):
        print(f'Y数据长度={len(Y)}')
        print(f'X数据长度={len(X)}')
        exit('回归数据Y和X长度不匹配')
    ###########################################
    # 主回归：得到残差
    ###########################################
    if nocons:
        model = sm.OLS(Y, X)
    else:
        model = sm.OLS(Y, sm.add_constant(X))
    fit = model.fit()
    e_hat_t = fit.resid
    ###########################################
    # 一阶滞后
    ###########################################
    e_hat_t_minus_1 = e_hat_t.shift(-1)
    # 用0填充NaN
    e_hat_t_minus_1 = np.nan_to_num(e_hat_t_minus_1, nan=0.0)

    Q = np.column_stack((X, e_hat_t_minus_1))
    if nocons:
        model = sm.OLS(e_hat_t, Q)
    else:
        model = sm.OLS(e_hat_t, sm.add_constant(Q))
    fit = model.fit()
    # 自由度为1的卡方分布
    v = fit.nobs * fit.rsquared
    print(f'LM({1})={fit.nobs * fit.rsquared :.3f} p={1 - stats.chi2.cdf(v, df=1) :.3f}')
    ###########################################
    # 二阶滞后
    ###########################################
    e_hat_t_minus_2 = pd.Series(e_hat_t_minus_1).shift(-1)
    # 用0填充NaN
    e_hat_t_minus_2 = np.nan_to_num(e_hat_t_minus_2, nan=0.0)

    R = np.column_stack((X, e_hat_t_minus_1, e_hat_t_minus_2))
    if nocons:
        model = sm.OLS(e_hat_t, R)
    else:
        model = sm.OLS(e_hat_t, sm.add_constant(R))
    fit = model.fit()
    # 自由度为2的卡方分布
    v = fit.nobs * fit.rsquared
    print(f'LM({2})={fit.nobs * fit.rsquared :.3f} p={1 - stats.chi2.cdf(v, df=2) :.3f}')
    ###########################################
    # 三阶滞后
    ###########################################
    e_hat_t_minus_3 = pd.Series(e_hat_t_minus_2).shift(-1)
    # 用0填充NaN
    e_hat_t_minus_3 = np.nan_to_num(e_hat_t_minus_3, nan=0.0)

    S = np.column_stack((X, e_hat_t_minus_1, e_hat_t_minus_2, e_hat_t_minus_3))
    if nocons:
        model = sm.OLS(e_hat_t, S)
    else:
        model = sm.OLS(e_hat_t, sm.add_constant(S))
    fit = model.fit(use_t=True)
    # 自由度为3的卡方分布
    v = fit.nobs * fit.rsquared
    print(f'LM({3})={fit.nobs * fit.rsquared :.3f} p={1 - stats.chi2.cdf(v, df=3) :.3f}')

# 回归方程是Y和X 最多3阶, 而且还显示出残差项的系数是否显著
# 可用于教材p163页
def LM_TEST_lag3_e_info(Y, X, nocons=False):

        print('LMTEST H0: 不存在序列相关性')
        if len(Y) != len(X):
            print(f'Y数据长度={len(Y)}')
            print(f'X数据长度={len(X)}')
            exit('回归数据Y和X长度不匹配')
        ###########################################
        # 主回归：得到残差
        ###########################################
        if nocons:
            model = sm.OLS(Y, X)
        else:
            model = sm.OLS(Y, sm.add_constant(X))
        fit = model.fit()
        e_hat_t = fit.resid
        ###########################################
        # 一阶滞后
        ###########################################
        e_hat_t_minus_1 = e_hat_t.shift(-1)
        # 用0填充NaN
        e_hat_t_minus_1 = np.nan_to_num(e_hat_t_minus_1, nan=0.0)

        Q = np.column_stack((X, e_hat_t_minus_1))
        if nocons:
            model = sm.OLS(e_hat_t, Q)
        else:
            model = sm.OLS(e_hat_t, sm.add_constant(Q))
        fit = model.fit(use_t=True)
        # 显示系数是否显著
        print('------残差1阶滞后项系数------')
        print(f'e({1})={fit.params[-1] :.3f} p={fit.pvalues[-1] :.3f}')
        # 自由度为1的卡方分布
        v = fit.nobs * fit.rsquared
        print(f'LM({1})={fit.nobs * fit.rsquared :.3f} p={1 - stats.chi2.cdf(v, df=1) :.3f}')
        ###########################################
        # 二阶滞后
        ###########################################
        e_hat_t_minus_2 = pd.Series(e_hat_t_minus_1).shift(-1)
        # 用0填充NaN
        e_hat_t_minus_2 = np.nan_to_num(e_hat_t_minus_2, nan=0.0)

        R = np.column_stack((X, e_hat_t_minus_1, e_hat_t_minus_2))
        if nocons:
            model = sm.OLS(e_hat_t, R)
        else:
            model = sm.OLS(e_hat_t, sm.add_constant(R))
        fit = model.fit(use_t=True)
        # 显示系数是否显著
        print('------残差2阶滞后项系数------')
        print(f'e({1})={fit.params[-2] :.3f} p={fit.pvalues[-2] :.3f}')
        print(f'e({2})={fit.params[-1] :.3f} p={fit.pvalues[-1] :.3f}')
        # 自由度为2的卡方分布
        v = fit.nobs * fit.rsquared
        print(f'LM({2})={fit.nobs * fit.rsquared :.3f} p={1 - stats.chi2.cdf(v, df=2) :.3f}')
        ###########################################
        # 三阶滞后
        ###########################################
        e_hat_t_minus_3 = pd.Series(e_hat_t_minus_2).shift(-1)
        # 用0填充NaN
        e_hat_t_minus_3 = np.nan_to_num(e_hat_t_minus_3, nan=0.0)

        S = np.column_stack((X, e_hat_t_minus_1, e_hat_t_minus_2, e_hat_t_minus_3))
        if nocons:
            model = sm.OLS(e_hat_t, S)
        else:
            model = sm.OLS(e_hat_t, sm.add_constant(S))
        fit = model.fit(use_t=True)
        # 显示系数是否显著
        print('------残差3阶滞后项系数------')
        print(f'e({1})={fit.params[-3] :.3f} p={fit.pvalues[-3] :.3f}')
        print(f'e({2})={fit.params[-2] :.3f} p={fit.pvalues[-2] :.3f}')
        print(f'e({3})={fit.params[-1] :.3f} p={fit.pvalues[-1] :.3f}')
        # 自由度为3的卡方分布
        v = fit.nobs * fit.rsquared
        print(f'LM({3})={fit.nobs * fit.rsquared :.3f} p={1 - stats.chi2.cdf(v, df=3) :.3f}')


# 回归方程是Y和X，可以是高阶
def LM_TEST_lagsN(Y, X, nocons, lags=3):
    '''
    是LM_TEST_lags3的升级版，可以做超过3阶
    :param Y:
    :param X:
    :param nocons:
    :param lags:
    :return:
    '''
    print('LMTEST H0: 不存在序列相关性')
    if len(Y) != len(X):
        print(f'Y数据长度={len(Y)}')
        print(f'X数据长度={len(X)}')
        exit('回归数据Y和X长度不匹配')
    ###########################################
    # 主回归：得到残差
    ###########################################
    if nocons == False:
        X = sm.add_constant(X)

    model = sm.OLS(Y, X)
    fit = model.fit()
    e_hat_t = fit.resid

    e_hat_t_series = [i for i in range(lags + 1)]
    e_hat_t_series[0] = e_hat_t


    for i in range(1, lags + 1):
        print(f'# {i}阶滞后')

        e_hat_t_series[i] = pd.Series(e_hat_t_series[i - 1]).shift(-1)
        # 用0填充NaN
        e_hat_t_series[i] = np.nan_to_num(e_hat_t_series[i], nan=0.0)
        S = X
        for j in range(1, i + 1):
            S = np.column_stack((S, e_hat_t_series[j]))
        model = sm.OLS(e_hat_t, S)
        fit = model.fit()

        v = fit.nobs * fit.rsquared
        print(f'LM({i})={fit.nobs * fit.rsquared :.3f} p={1 - stats.chi2.cdf(v, df=i) :.3f}')
    print('###########################################')

def ADF1_LM_TEST_lags(Y, MODEL=3, lags=3):
    '''
    ADF1 表示 Y滞后1阶
    序列Y平稳性中的LM test函数
    回归方程是Y自身，可以是高阶
    其中 ADF的参数是MODEL
    LM_TEST的参数是LAGS
    :param Y: 滞后一阶代入回归方程中
    :param MODEL: ADF的参数，选3个方程中的哪一个
    :param lags: LM_TEST的参数相关性滞后几阶
    :return:
    '''

    Y_d1 = Y - Y.shift(-1)
    Y_lag1 = Y.shift(-1)
    Y_d1_lag1 = Y_d1.shift(-1)

    # 时间是从近到远
    # Y数据也是从最近到以前
    mylist = [i for i in range(Y.size-1, -1, -1)]
    T = pd.DataFrame(data=mylist)

    if MODEL == 3:
        # 模型3：常数项+时间趋势项+Y滞后项
        X = np.column_stack((T, Y_lag1, Y_d1_lag1))

        # 对整个数组删除有NaN值的每一行
        X = X[~np.any(np.isnan(X), axis=1)]  # 1 按行； 0 按列
        X = sm.add_constant(X)

    elif MODEL == 2:
        # 模型2：常数项+Y滞后项
        X = np.column_stack((Y_lag1, Y_d1_lag1))
        X = X[~np.any(np.isnan(X), axis=1)]  # 1 按行； 0 按列
        X = sm.add_constant(X)

    elif MODEL == 1:
        # 模型1：没有常数项和时间趋势项，仅有Y滞后项
        X = np.column_stack((Y_lag1, Y_d1_lag1))
        X = X[~np.any(np.isnan(X), axis=1)]  # 1 按行； 0 按列

    else:
        exit('MODEL=3 # INTERCEPTION + TIME TREND \n'
             'MODEL=2 # NO TIME TREND \n'
             'MODEL=1 # NO CONST')

    Y_d1 = Y_d1[:-2]
    if len(Y_d1) != len(X):
        print(f'Y_d1数据长度={len(Y_d1)}')
        print(f'X数据长度={len(X)}')
        exit('回归数据Y_d1和X长度不匹配')
    ###########################################
    # 主回归：得到残差
    ###########################################

    model = sm.OLS(Y_d1, X)
    fit = model.fit()
    e_hat_t = fit.resid

    e_hat_t_series = [i for i in range(lags+1)]
    e_hat_t_series[0] = e_hat_t


    print(f'############ MODEL = {MODEL} ##############')
    for i in range(1, lags+1):

        print(f'# {i}阶滞后')

        e_hat_t_series[i] = pd.Series(e_hat_t_series[i-1]).shift(-1)
        # 用0填充NaN
        e_hat_t_series[i] = np.nan_to_num(e_hat_t_series[i], nan=0.0)
        S = X
        for j in range(1, i+1):
            S = np.column_stack((S, e_hat_t_series[j]))
        model = sm.OLS(e_hat_t, S)
        fit = model.fit()

        v = fit.nobs * fit.rsquared
        print(f'LM({i})={fit.nobs * fit.rsquared :.3f} p={1 - stats.chi2.cdf(v, df=i) :.3f}')
    print('###########################################')



