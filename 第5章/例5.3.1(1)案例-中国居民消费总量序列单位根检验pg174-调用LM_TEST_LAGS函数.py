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

Y = df.iloc[:, 6].astype(np.float64)

mylmtest.LM_TEST_lags(Y, MODEL=3, lags=3)  # INTERCEPTION + TIME TREND
mylmtest.LM_TEST_lags(Y, MODEL=2, lags=3)  # NO TIME TREND
mylmtest.LM_TEST_lags(Y, MODEL=1, lags=3)  # NO CONST
