# Author: 史浩 浙江金融职业学院
# --------------------------------------------------------
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv(r'data\data1.csv', encoding='gbk')
print(df)
df = df.iloc[4:35, :3]
df.columns = ['province', 'X', 'Y']
df.reset_index(drop=True, inplace=True)  # 把索引重新排一下
print(df)
X = sm.add_constant(df.iloc[:, 1])
print(X)
# 用.astype(float)将EXCEL文件中的字符型转浮点型
model = sm.OLS(df.iloc[:, 2].astype(float), X.astype(float))
result = model.fit().summary()
print(result)
