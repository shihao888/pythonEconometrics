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
GDP = df.iloc[:, 1].astype(float)
CPI = df.iloc[:, 3].astype(float)
GDPR = GDP / CPI * 100


from statsmodels.tsa.stattools import grangercausalitytests

Q = np.column_stack((GDPR, Y))
Q = Q[~np.any(np.isnan(Q) | np.equal(Q, 0), axis=1)]  # 1 按行； 0 按列

if len(GDPR) != len(Y):
    print(f'GDPR数据长度={len(GDPR)}')
    print(f'Y数据长度={len(Y)}')
    exit('回归数据GDPR和Y长度不匹配')

df = pd.DataFrame(Q, columns=['GDPR', 'Y'])


# 主要看p值（第二列），所有的p小于0.05才能证明有效
# 结论： a格兰杰导致b  GX 导致 GY

grangercausalitytests(df[['GDPR', 'Y']], maxlag=4, verbose=True)
grangercausalitytests(df[['Y', 'GDPR']], maxlag=4, verbose=True)