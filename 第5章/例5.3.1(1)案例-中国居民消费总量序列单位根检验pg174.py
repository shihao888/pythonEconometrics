# --------------------------------------------------------
# Author: 史浩 浙江金融职业学院
# --------------------------------------------------------
# coding=utf-8
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

df.sort_values(by=['year'], ascending=True, inplace=True)  # 时间序列按升序排
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

T = df.iloc[:, 0].astype(int)
Y = df.iloc[:, 6].astype(float)

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


#  CTRL+SHIFT+I to check the function help
Y_d1 = Y - Y.shift(-1)
Y_lag1 = Y.shift(-1)
Y_d1_lag1 = Y_d1.shift(-1)

X = np.column_stack((T, Y_lag1, Y_d1_lag1))
X = X.dropna()
##########################################################################
# 进行多元线性回归，可以替换Y和X
#          OLS：  Y          = beta0 + beta1*X1 + beta2*X2
##########################################################################
model = sm.OLS(Y_d1, sm.add_constant(X))  # 用add_constant加入常数项
fit = model.fit(use_t=True)  # not using HC0,HC1 etc.
print(fit.summary(title='Y_d1', yname='Y_d1',
                  xname=['const', 'T', 'Y_lag1', 'Y_d1_lag1']))


# 添加一项滞后项，再回归，再检验
# lnY_d1 = lnY.shift(-1)
# model = sm.OLS(lnY, sm.add_constant(lnX))  # 用add_constant加入常数项
# fit = model.fit(use_t=True)  # not using HC0,HC1 etc.
# print(fit.summary(yname='lnY', xname=['const', 'lnX']))
