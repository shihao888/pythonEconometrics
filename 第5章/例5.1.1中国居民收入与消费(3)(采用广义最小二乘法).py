# --------------------------------------------------------
# Author: 史浩 浙江金融职业学院
# --------------------------------------------------------
# coding=utf-8
import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_excel(r'..\data\例题数据.xlsx', sheet_name="5.1.1")
df = df.iloc[2:, :7]  # 注意：第0行是表头，要从第1行开始获得数据
df = df.dropna()
# print(df)
#              0      1       2      3      4      5    6
df.columns = ['year', 'GDP', 'CONS', 'CPI', 'TAX', 'X', 'Y']

df.sort_values(by=['year'], ascending=False, inplace=True)  # 时间序列按降序排
df.reset_index(drop=True, inplace=True)  # 把索引重新排一下
# print(df)

# 定义变量
# 用.astype(float)将EXCEL文件中的字符型转浮点型
Y = df.iloc[:, 6].astype(float)
X = df.iloc[:, 5].astype(float)
##########################################################################
# 进行多元线性回归，可以替换Y和X
#          OLS：  Y          = beta0 + beta1*X1 + beta2*X2
##########################################################################
model = sm.OLS(Y, sm.add_constant(X))  # 用add_constant加入常数项
fit = model.fit(use_t=True)  # not using HC0,HC1 etc.
# print(fit.summary())

# 一阶滞后
e_hat_t = fit.resid
e_hat_t_minus_1 = e_hat_t.shift(-1)
e_hat_t_minus_1 = e_hat_t_minus_1.iloc[:-1]  # 丢弃最后一行N/A
e_hat_t = e_hat_t.iloc[:-1]  # 为保持数据一致，也丢弃最后一行

model = sm.OLS(e_hat_t, e_hat_t_minus_1)
fit = model.fit(use_t=True)
print(fit.summary(title='find Rho1', yname='e_hat_t', xname=['rho1']))
rho1 = fit.params[0]
print(f'rho1={rho1}')


# # 二阶滞后
# e_hat_t_minus_2 = e_hat_t_minus_1.shift(-1)
# e_hat_t_minus_2 = e_hat_t_minus_2.iloc[:-1]  # 之前丢过一行，在之前基础上再多丢一行
#
# e_hat_t = e_hat_t.iloc[:-1]  # 为保持数据一致，也丢弃最后两行(之前丢过一行，在之前基础上再多丢一行)
# e_hat_t_minus_1 = e_hat_t_minus_1.iloc[:-1]
#
# R = np.column_stack((e_hat_t_minus_1, e_hat_t_minus_2))
# model = sm.OLS(e_hat_t, R)
# fit = model.fit(use_t=True)
# print(fit.summary(yname='e_hat_t', xname=['rho1', 'rho2']))
# 由于二阶滞后不显著，因此只存在一阶滞后。
def get_rho(rho1):
    Y = df.iloc[:, 6].astype(float)
    X = df.iloc[:, 5].astype(float)

    Y_star = Y - rho1 * Y.shift(-1)
    X_star = X - rho1 * X.shift(-1)

    # 假设不补足
    Y_star = Y_star.iloc[:-1]
    X_star = X_star.iloc[:-1]
    model = sm.OLS(Y_star, sm.add_constant(X_star))
    fit = model.fit(use_t=True)
    # print(fit.summary())
    rho1 = fit.params[0]

    # 代入原回归方程得到新的残差
    beta0 = fit.params[0] / (1 - rho1)
    beta1 = fit.params[1]
    new_e_hat_t = Y - (beta0 + beta1 * X)
    new_e_hat_t_minus_1 = new_e_hat_t.shift(-1)
    new_e_hat_t_minus_1 = new_e_hat_t_minus_1.iloc[:-1]  # 丢弃最后一行N/A
    new_e_hat_t = new_e_hat_t.iloc[:-1]  # 为保持数据一致，也丢弃最后一行

    model = sm.OLS(new_e_hat_t, new_e_hat_t_minus_1)
    fit = model.fit(use_t=True)
    # print(fit.summary(title='find new_Rho1', yname='new_e_hat_t', xname=['rho1']))
    rho1 = fit.params[0]
    return rho1


# step1.广义最小二乘法
Y = df.iloc[:, 6].astype(float)
X = df.iloc[:, 5].astype(float)

Y_star = Y - rho1 * Y.shift(-1)
X_star = X - rho1 * X.shift(-1)

print("=" * 50)
print("Prais-Winsten transformation:")
print("=" * 50)
print(f'补足前Y_star.iloc[-1]={Y_star.iloc[-1]}')
# step2.Prais-Winsten transformation
# 补足缺失的最后一年的数据s
Y_star.iloc[-1] = Y.iloc[-1] * np.sqrt(1 - rho1 ** 2)
X_star.iloc[-1] = X.iloc[-1] * np.sqrt(1 - rho1 ** 2)
print(f'补足后Y_star.iloc[-1]={Y_star.iloc[-1]}')
# print(Y_star)
# print(X_star)
model = sm.OLS(Y_star, sm.add_constant(X_star))  # 用add_constant加入常数项
fit = model.fit(use_t=True)  # not using HC0,HC1 etc.
print(fit.summary(yname='Y_star', xname=['const', 'X_star']))

# 假设不补足
Y_star = Y_star.iloc[:-1]
X_star = X_star.iloc[:-1]
model = sm.OLS(Y_star, sm.add_constant(X_star))
fit = model.fit(use_t=True)
print(fit.summary())

print(get_rho(rho1))
print(get_rho(rho1))
# Newey-West标准误(异方差自相关稳健的标准误)
# cov_type='HAC' cov_kwds={'maxlags': 1}
model = sm.OLS(Y, sm.add_constant(X))
fit = model.fit(cov_type='HAC', use_T=True, cov_kwds={'maxlags': 1})
print(fit.summary())
