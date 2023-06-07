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
# 异方差稳健的标准误 ：
#     HC0:White（1980）提出的异方差稳健的标准误
#     HC1:Mackinon and White（1985）提出的异方差稳健的标准误
#     HC2:MacKinnon and White（1985）提出的异方差稳健的标准误
#     HC3:MacKinnon and White（1985）提出的异方差稳健的标准误
#     HAC:Newey-West标准误（异方差自相关稳健的标准误）
fit = model.fit(use_t=True)  # not using HC0,HC1 etc.
print(fit.summary(title='基本回归'))

# 一阶滞后
e_hat_t = fit.resid
e_hat_t_minus_1 = e_hat_t.shift(-1)
e_hat_t_minus_1 = e_hat_t_minus_1.iloc[:-1]  # 丢弃最后一行N/A
e_hat_t = e_hat_t.iloc[:-1]  # 为保持数据一致，也丢弃最后一行
X = X.iloc[:-1]  # 为保持数据一致，也丢弃最后一行

Q = np.column_stack((X, e_hat_t_minus_1))
model = sm.OLS(e_hat_t, sm.add_constant(Q))
fit = model.fit(use_t=True)
print(fit.summary(title='残差项回归', yname='残差', xname=['常数项', 'X', '残差一阶滞后项']))

# 二阶滞后
e_hat_t_minus_2 = e_hat_t_minus_1.shift(-1)
e_hat_t_minus_2 = e_hat_t_minus_2.iloc[:-1]  # 之前丢过一行，在之前基础上再多丢一行

e_hat_t = e_hat_t.iloc[:-1]  # 为保持数据一致，也丢弃最后两行(之前丢过一行，在之前基础上再多丢一行)
e_hat_t_minus_1 = e_hat_t_minus_1.iloc[:-1]
X = X.iloc[:-1]

R = np.column_stack((X, e_hat_t_minus_1, e_hat_t_minus_2))
model = sm.OLS(e_hat_t, sm.add_constant(R))
fit = model.fit(use_t=True)
print(fit.summary(title='残差项回归', yname='残差', xname=['常数项', 'X', '残差一阶滞后项', '残差二阶阶滞后项']))
