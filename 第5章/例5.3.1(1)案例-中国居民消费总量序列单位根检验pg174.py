# --------------------------------------------------------
# Author: 史浩 浙江金融职业学院
# --------------------------------------------------------
# coding=utf-8
import 第5章.LM_TEST as mylmtest
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.tsa.stattools import kpss


df = pd.read_excel(r'..\data\例题数据.xlsx', sheet_name="5.1.1")
df = df.iloc[2:, :7]  # 注意：第0行是表头，要从第1行开始获得数据
df = df.dropna()
# print(df)
#              0      1       2      3      4      5    6
df.columns = ['year', 'GDP', 'CONS', 'CPI', 'TAX', 'X', 'Y']

df.sort_values(by=['year'], ascending=False, inplace=True)  # 时间序列按降序排
df.reset_index(drop=True, inplace=True)  # 把索引重新排一下

# 定义变量
# 用.astype(float)将EXCEL文件中的字符型转浮点型
Y = df.iloc[:, 6].astype(float)

# LM检验自相关性
# acorr_lm将会包含四个变量：LM统计量、LM统计量的p值、F统计量与F统计量的p值
# Conduct the lm test
print('='*50)
print('自相关检验 H0:不存在自相关')
lm_test = sms.acorr_lm(Y, nlags=1)  # fit.model.exog 就是 lnX
print('Lagrange multiplier statistic: %f' % lm_test[0])
print('The p value: %f' % lm_test[1])

lb_test = sm.stats.acorr_ljungbox(Y, lags=10)
print('The Ljung-Box test statistic:')
print(lb_test)

T = df.iloc[:, 0].astype(int) - 1978
Y = df.iloc[:, 6].astype(np.float64)

# ADF检验 平稳性检验
# H0：具有单位根，属于非平稳序列。
from statsmodels.tsa.stattools import adfuller

'''
regression : {"c","ct","ctt","n"}
        Constant and trend order to include in regression.
* "c" : constant only (default).
* "ct" : constant and trend.
* "ctt" : constant, and linear and quadratic trend.
* "n" : no constant, no trend.
'''
print('='*50)
print('平稳性检验 H0：具有单位根，非平稳序列')
adf_result = adfuller(Y, regression='n')  # 生成adf检验结果, c ct ctt n
print('The ADF Statistic: %f' % adf_result[0])
print('The p value: %f' % adf_result[1])

# 模型3：常数项+时间趋势项+Y滞后项
#  CTRL+SHIFT+I to check the function help
Y_d1 = Y - Y.shift(-1)
Y_lag1 = Y.shift(-1)
Y_d1_lag1 = Y_d1.shift(-1)

X = np.column_stack((T, Y_lag1, Y_d1_lag1))
# 对整个数组删除有NaN值的每一行
X = X[~np.any(np.isnan(X), axis=1)]  # 1 按行； 0 按列
Y_d1 = Y_d1.iloc[:-2]
if len(Y_d1) != len(X):
    print(f'Y数据长度={len(Y_d1)}')
    print(f'X数据长度={len(X)}')
    exit('回归数据Y和X长度不匹配')
##########################################################################
# 进行多元线性回归，可以替换Y和X
#          OLS：  Y          = beta0 + beta1*X1 + beta2*X2
##########################################################################
model = sm.OLS(Y_d1, sm.add_constant(X))  # 用add_constant加入常数项
fit = model.fit(use_t=True)  # not using HC0,HC1 etc.
print(fit.summary(title='Y_d1', yname='Y_d1',
                  xname=['const', 'T', 'Y_lag1', 'Y_d1_lag1']))
print('model3 H0: 存在单位根。请查看Y_lag1的显著性看是否拒绝H0')
mylmtest.LM_TEST(Y_d1, X)

# 模型2：常数项+Y滞后项
X = np.column_stack((Y_lag1, Y_d1_lag1))
X = X[~np.any(np.isnan(X), axis=1)]  # 1 按行； 0 按列
model = sm.OLS(Y_d1, sm.add_constant(X))  # 用add_constant加入常数项
fit = model.fit(use_t=True)  # not using HC0,HC1 etc.
print(fit.summary(title='Y_d1', yname='Y_d1',
                  xname=['const', 'Y_lag1', 'Y_d1_lag1']))
print('model2 H0: 存在单位根。请查看Y_lag1的显著性看是否拒绝H0')
mylmtest.LM_TEST(Y_d1, X)

# 模型1：没有常数项和时间趋势项，仅有Y滞后项
X = np.column_stack((Y_lag1, Y_d1_lag1))
X = X[~np.any(np.isnan(X), axis=1)]  # 1 按行； 0 按列
model = sm.OLS(Y_d1, X)
fit = model.fit(use_t=True)  # not using HC0,HC1 etc.
print(fit.summary(title='Y_d1', yname='Y_d1',
                  xname=['Y_lag1', 'Y_d1_lag1']))
print('model1 H0: 存在单位根。请查看Y_lag1的显著性看是否拒绝H0')
mylmtest.LM_TEST(Y_d1, X)

