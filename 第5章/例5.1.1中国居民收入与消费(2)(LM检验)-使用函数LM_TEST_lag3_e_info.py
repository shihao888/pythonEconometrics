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


# 确定滞后阶数
import 第5章.LM_TEST as mylmtest


Y = df.iloc[:, 6].astype(float)
X = df.iloc[:, 5].astype(float)

# Y一定要按照时间降序排列（时间从近到远）才可以用这个函数LM_TEST_lag3
mylmtest.LM_TEST_lag3_e_info(Y, X, nocons=False)
