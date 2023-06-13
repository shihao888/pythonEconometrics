# Author: 史浩 浙江金融职业学院
# --------------------------------------------------------
import pandas as pd
import statsmodels.api as sm

df = pd.read_excel(r'data\myexcel.xlsx')
X = sm.add_constant(df.iloc[:, 1])
print(X)
model = sm.OLS(df.iloc[:, 0], X)
result = model.fit().summary()
print(result)
