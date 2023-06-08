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
T = df.iloc[:, 0].astype(int)
Y = df.iloc[:, 6].astype(float)


# 模型3：常数项+时间趋势项+Y滞后项
#  CTRL+SHIFT+I to check the function help
Y_d1 = Y - Y.shift(-1)
Y_d2 = Y_d1 - Y_d1.shift(-1)
Y_d3 = Y_d2 - Y_d2.shift(-1)
Y_d1_lag1 = Y_d1.shift(-1)
Y_d2_lag1 = Y_d2.shift(-1)
Y_d3_lag1 = Y_d3.shift(-1)
Y_d3_lag2 = Y_d3_lag1.shift(-1)

X = np.column_stack((T, Y_d2_lag1, Y_d3_lag1, Y_d3_lag2))
# 对整个数组删除有NaN值的每一行
X = X[~np.any(np.isnan(X), axis=1)]  # 1 按行； 0 按列
Y_d3 = Y_d3.iloc[:-5]
if len(Y_d3) != len(X):
    print(f'Y数据长度={len(Y_d3)}')
    print(f'X数据长度={len(X)}')
    exit('回归数据Y和X长度不匹配')
##########################################################################
# 进行多元线性回归，可以替换Y和X
#          OLS：  Y          = beta0 + beta1*X1 + beta2*X2
##########################################################################
model = sm.OLS(Y_d3, sm.add_constant(X))  # 用add_constant加入常数项
fit = model.fit()  # not using HC0,HC1 etc.
print(fit.summary(title='Y_d2', yname='Y_d2',
                  xname=['const', 'T', 'Y_d2_lag1', 'Y_d3_lag1', 'Y_d3_lag2']))
print('model3 H0: 存在单位根。请查看Y_d2_lag1的显著性看是否拒绝H0')
mylmtest.LM_TEST(Y_d3, X)

# 模型2：常数项+Y滞后项
X = Y_d2_lag1
X = X.iloc[:-5]
# X = X[~np.any(np.isnan(X), axis=1)]  # 1 按行； 0 按列
if len(Y_d3) != len(X):
    print(f'Y数据长度={len(Y_d3)}')
    print(f'X数据长度={len(X)}')
    exit('回归数据Y和X长度不匹配')
model = sm.OLS(Y_d3, sm.add_constant(X))  # 用add_constant加入常数项
fit = model.fit(use_t=True)  # not using HC0,HC1 etc.
print(fit.summary(title='Y_d3', yname='Y_d3',
                  xname=['const', 'Y_d2_lag1']))
print('model2 H0: 存在单位根。请查看Y_d1_lag1的显著性看是否拒绝H0')
mylmtest.LM_TEST(Y_d3, X)

# 模型1：没有常数项和时间趋势项，仅有Y滞后项
X = Y_d2_lag1
X = X.iloc[:-5]
# X = X[~np.any(np.isnan(X), axis=1)]  # 1 按行； 0 按列
model = sm.OLS(Y_d3, X)
fit = model.fit(use_t=True)  # not using HC0,HC1 etc.
print(fit.summary(title='Y_d3', yname='Y_d3',
                  xname=['Y_d2_lag1']))
print('model1 H0: 存在单位根。请查看Y_d1_lag1的显著性看是否拒绝H0')
mylmtest.LM_TEST(Y_d3, X)

