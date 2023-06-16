# --------------------------------------------------------
# Author: 史浩 浙江金融职业学院
# --------------------------------------------------------
# coding=utf-8
# --------------------------------------------------------

import 第5章.LM_TEST as mylmtest
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats


def ADF_LM_TEST_lags(Y, MODEL=3, adf_lags=1, lmlags=3):
    """
    ADF1 表示 Y滞后1阶
    序列Y平稳性中的LM test函数
    回归方程是Y自身，可以是高阶
    其中 ADF的参数是MODEL
    LM_TEST的参数是LAGS
    :param Y: 滞后一阶代入回归方程中
    :param MODEL: ADF的参数，选3个方程中的哪一个
    :param adf_lags: ADF_TEST的参数相关性滞后几阶
    :param lmlags: LM_TEST的参数相关性滞后几阶
    :return:
    """

    Y_d1 = Y - Y.shift(-1)
    Y_lag1 = Y.shift(-1)

    Y_d1_series = [i for i in range(adf_lags + 1)]
    Y_d1_series[0] = Y_d1
    for k in range(1, adf_lags + 1):
        Y_d1_series[k] = pd.Series(Y_d1_series[k - 1]).shift(-1)
        # 用0填充NaN
        Y_d1_series[k] = np.nan_to_num(Y_d1_series[k], nan=0.0)

    Y_d1_family = Y_d1_series[1]

    # 时间是从近到远
    # Y数据也是从最近到以前
    mylist = [i for i in range(Y.size - 1, -1, -1)]
    T = pd.DataFrame(data=mylist)

    if MODEL == 3:
        # 模型3：常数项+时间趋势项+Y滞后项
        for k in range(2, adf_lags + 1):
            Y_d1_family = np.column_stack((Y_d1_family, Y_d1_series[k]))
        X = np.column_stack((T, Y_lag1, Y_d1_family))

        # 对整个数组删除有NaN值的每一行
        X = X[~np.any(np.isnan(X), axis=1)]  # 1 按行； 0 按列
        X = sm.add_constant(X)

    elif MODEL == 2:
        # 模型2：常数项+Y滞后项
        for k in range(2, adf_lags + 1):
            Y_d1_family = np.column_stack((Y_d1_family, Y_d1_series[k]))
        X = np.column_stack((Y_lag1, Y_d1_family))
        X = X[~np.any(np.isnan(X), axis=1)]  # 1 按行； 0 按列
        X = sm.add_constant(X)

    elif MODEL == 1:
        # 模型1：没有常数项和时间趋势项，仅有Y滞后项
        for k in range(2, adf_lags + 1):
            Y_d1_family = np.column_stack((Y_d1_family, Y_d1_series[k]))
        X = np.column_stack((Y_lag1, Y_d1_family))
        X = X[~np.any(np.isnan(X), axis=1)]  # 1 按行； 0 按列

    else:
        exit('MODEL=3 # INTERCEPTION + TIME TREND \n'
             'MODEL=2 # NO TIME TREND \n'
             'MODEL=1 # NO CONST')

    Y_d1 = Y_d1[:-1]
    if len(Y_d1) != len(X):
        print(f'Y_d1数据长度={len(Y_d1)}')
        print(f'X数据长度={len(X)}')
        exit('回归数据Y_d1和X长度不匹配')
    ###########################################
    # 主回归：得到残差
    ###########################################

    model = sm.OLS(Y_d1, X)
    fit = model.fit()
    print(fit.summary(title=f"=========ADF中Delta-Y滞后{adf_lags}阶========="))
    e_hat_t = fit.resid

    e_hat_t_series = [i for i in range(lmlags + 1)]
    e_hat_t_series[0] = e_hat_t

    print(f'############ MODEL = {MODEL} ##############')
    for i in range(1, lmlags + 1):

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


def ADF_LM_TEST_lags(Y, MODEL=3, adf_lags=1):
    """
    ADF1 表示 Y滞后1阶
    序列Y平稳性中的LM test函数
    回归方程是Y自身，可以是高阶
    其中 ADF的参数是MODEL
    LM_TEST的参数是LAGS
    :param Y: 滞后一阶代入回归方程中
    :param MODEL: ADF的参数，选3个方程中的哪一个
    :param adf_lags: ADF_TEST的参数相关性滞后几阶
    :return:
    """

    Y_d1 = Y - Y.shift(-1)
    Y_lag1 = Y.shift(-1)

    Y_d1_series = [i for i in range(adf_lags + 1)]
    Y_d1_series[0] = Y_d1
    for k in range(1, adf_lags + 1):
        Y_d1_series[k] = pd.Series(Y_d1_series[k - 1]).shift(-1)
        # 用0填充NaN
        Y_d1_series[k] = np.nan_to_num(Y_d1_series[k], nan=0.0)

    Y_d1_family = Y_d1_series[1]

    # 时间是从近到远
    # Y数据也是从最近到以前
    mylist = [i for i in range(Y.size - 1, -1, -1)]
    T = pd.DataFrame(data=mylist)

    if MODEL == 3:
        # 模型3：常数项+时间趋势项+Y滞后项
        for k in range(2, adf_lags + 1):
            Y_d1_family = np.column_stack((Y_d1_family, Y_d1_series[k]))
        X = np.column_stack((T, Y_lag1, Y_d1_family))

        # 对整个数组删除有NaN值的每一行
        X = X[~np.any(np.isnan(X) | np.equal(X, 0), axis=1)]  # 1 按行； 0 按列
        X = sm.add_constant(X)

    elif MODEL == 2:
        # 模型2：常数项+Y滞后项
        for k in range(2, adf_lags + 1):
            Y_d1_family = np.column_stack((Y_d1_family, Y_d1_series[k]))
        X = np.column_stack((Y_lag1, Y_d1_family))
        X = X[~np.any(np.isnan(X) | np.equal(X, 0), axis=1)]  # 1 按行； 0 按列
        X = sm.add_constant(X)

    elif MODEL == 1:
        # 模型1：没有常数项和时间趋势项，仅有Y滞后项
        for k in range(2, adf_lags + 1):
            Y_d1_family = np.column_stack((Y_d1_family, Y_d1_series[k]))
        X = np.column_stack((Y_lag1, Y_d1_family))
        X = X[~np.any(np.isnan(X) | np.equal(X, 0), axis=1)]  # 1 按行； 0 按列

    else:
        exit('MODEL=3 # INTERCEPTION + TIME TREND \n'
             'MODEL=2 # NO TIME TREND \n'
             'MODEL=1 # NO CONST')

    Y_d1 = Y_d1[:-1-adf_lags]
    if len(Y_d1) != len(X):
        print(f'Y_d1数据长度={len(Y_d1)}')
        print(f'X数据长度={len(X)}')
        exit('回归数据Y_d1和X长度不匹配')
    ###########################################
    # 主回归：得到残差
    ###########################################

    model = sm.OLS(Y_d1, X)
    fit = model.fit()
    print(fit.summary(title=f"=========ADF中Delta-Y滞后{adf_lags}阶========="))

