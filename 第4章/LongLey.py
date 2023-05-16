# Author: 史浩 浙江金融职业学院
# Multicollinearity
# Longley数据集来自J．W．Longley（1967）发表在JASA上的一篇论文，
# 是强共线性的宏观经济数据,包含GNP deflator(GNP平减指数)、GNP(国民生产总值)、
# Unemployed(失业率)、rmedForces(武装力量)、Population(人口)、year(年份)，
# Emlpoyed(就业率)。
# LongLey数据集因存在严重的多重共线性问题，在早期经常用来检验各种算法或计算机的计算精度。
# 原文链接：https://blog.csdn.net/JerryZhang1111/article/details/116497162
# --------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

np.random.seed(9876789)

from statsmodels.datasets.longley import load_pandas

y = load_pandas().endog
X = load_pandas().exog
X = sm.add_constant(X)


ols_model = sm.OLS(y, X)
ols_results = ols_model.fit()
print(ols_results.summary())


