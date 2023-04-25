# Author: 史浩 浙江金融职业学院
# --------------------------------------------------------
import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_excel(r'..\data\例题数据.xlsx', sheet_name="3.6.1")
df = df.iloc[:, :9]
df = df.dropna()
# print(df)
#              0      1    2     3     4     5    6     7     8
df.columns = ['area', 'Y', 'X1', 'X2', 'X3', 'Y', 'X1', 'X2', 'X3']
df.reset_index(drop=True, inplace=True)  # 把索引重新排一下
# print(df)
# 农村
Y = df.iloc[:-1, 1].astype(float)
X1 = df.iloc[:-1, 2].astype(float)
X2 = df.iloc[:-1, 3].astype(float)
X3 = df.iloc[:-1, 4].astype(float)
Z = np.column_stack((X1, X2, X3))
##########################################################################
# 进行多元线性回归，可以替换Y和X
#          OLS：  Y          = beta0 + beta1*X1 + beta2*X2
##########################################################################
model = sm.OLS(Y, sm.add_constant(Z))  # 用add_constant加入常数项
print(model.fit().summary(xname=['const', 'X1', 'X2', 'X3']))  # 用自己的名称命名常数和各个解释变量

# 城市
Y = df.iloc[:-1, 5].astype(float)
X1 = df.iloc[:-1, 6].astype(float)
X2 = df.iloc[:-1, 7].astype(float)
X3 = df.iloc[:-1, 8].astype(float)
Z = np.column_stack((X1, X2, X3))
##########################################################################
# 进行多元线性回归，可以替换Y和X
#          OLS：  Y          = beta0 + beta1*X1 + beta2*X2
##########################################################################
model = sm.OLS(Y, sm.add_constant(Z))  # 用add_constant加入常数项
print(model.fit().summary(xname=['const', 'X1', 'X2', 'X3']))  # 用自己的名称命名常数和各个解释变量

# 农村+城市
Y = pd.concat([df.iloc[:-1, 1].astype(float), df.iloc[:-1, 5].astype(float)]).reset_index(drop=True)
X1 = pd.concat([df.iloc[:-1, 2].astype(float), df.iloc[:-1, 6].astype(float)]).reset_index(drop=True)
X2 = pd.concat([df.iloc[:-1, 3].astype(float), df.iloc[:-1, 7].astype(float)]).reset_index(drop=True)
X3 = pd.concat([df.iloc[:-1, 4].astype(float), df.iloc[:-1, 8].astype(float)]).reset_index(drop=True)
Z = np.column_stack((X1, X2, X3))
##########################################################################
# 进行多元线性回归，可以替换Y和X
#          OLS：  Y          = beta0 + beta1*X1 + beta2*X2
##########################################################################
model = sm.OLS(Y, sm.add_constant(Z))  # 用add_constant加入常数项
print(model.fit().summary(xname=['const', 'X1', 'X2', 'X3']))  # 用自己的名称命名常数和各个解释变量


def evaluateModel(themodel):
    # e = Y - themodel.fit().predict()
    RSS = ((Y - themodel.fit().predict()) ** 2).sum()
    ESS = ((themodel.fit().predict()) ** 2).sum()
    print("\n\n"+"="*30+"Other Statistics"+"="*30)
    print(f"RSS = {RSS:.6f}")
    print(f"ESS = {ESS:.6f}")
    print(f"TSS = {(Y ** 2).sum():.6f}")
    print(f"TSS = {RSS + ESS:.6f}")
    print(f"R2 = {themodel.fit().rsquared:.6f}")
    print(f"adjR2 = {themodel.fit().rsquared_adj:.6f}")
    print("=" * 30 + "Other Statistics" + "=" * 30)
    return


evaluateModel(model)
