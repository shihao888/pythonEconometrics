# Author: 史浩 浙江金融职业学院
# --------------------------------------------------------
import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_excel(r'..\data\例题数据.xlsx', sheet_name="4.1.2")
df = df.iloc[:, :8]
df = df.dropna()
# print(df)
#              0      1     2    3     4     5     6     7
df.columns = ['area', 'Y', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6']
df.reset_index(drop=True, inplace=True)  # 把索引重新排一下
# print(df)

# 定义变量
# 用.astype(float)将EXCEL文件中的字符型转浮点型
Y = df.iloc[:, 1].astype(float).apply(np.log)
X1 = df.iloc[:, 2].astype(float).apply(np.log)
X2 = df.iloc[:, 3].astype(float).apply(np.log)
X3 = df.iloc[:, 4].astype(float).apply(np.log)
X4 = df.iloc[:, 5].astype(float).apply(np.log)
X5 = df.iloc[:, 6].astype(float).apply(np.log)
X6 = df.iloc[:, 7].astype(float).apply(np.log)
X = np.column_stack((X1, X2, X3, X4, X5, X6))
##########################################################################
# 进行多元线性回归，可以替换Y和X
#          OLS：  Y          = beta0 + beta1*X1 + beta2*X2
##########################################################################
model = sm.OLS(Y, sm.add_constant(X))  # 用add_constant加入常数项
print(model.fit().summary())
print("R2太高，而且F值太大，说明有多重共线性。")
print("检验其相关系数：")
# 将多个列合并成一个表，也就是将多个Series对象合并成一个DataFrame对象。
# 主要推荐的函数有pd.concat()
# 和 pd.DataFrame(list(zip(s1, s2, s3)))
X_df = pd.concat([X1, X2, X3, X4, X5, X6], axis=1, ignore_index=True)  # 列合并
# 相关系数：X_df.corr返回结果是一个数据框corr_df，存放的是相关系数矩阵corr_df
corr_df = X_df.corr()
print('相关系数矩阵：')
print(corr_df)

# 可利用逐步回归筛选并剔除引起多重共线性的变量，
# 其具体步骤如下：
# 先用被解释变量对每一个所考虑的解释变量做简单回归，
# 然后以对被解释变量贡献最大的解释变量所对应的回归方程为基础，
# 再逐步引入其余解释变量。经过逐步回归，
# 使得最后保留在模型中的解释变量既是重要的，又没有严重多重共线性。
# 导入要用到的各种包和函数

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.api import anova_lm
import matplotlib.pyplot as plt
import pandas as pd
import itertools as it
import random

# df中所有的数据都取对数
df = df.iloc[1:, 1:].astype(float).apply(np.log)
print(df.columns)
variate = list(df.columns)  # 获取列名
variate.remove('Y')  # 去除无关列

# 定义多个数组，用来分别用来添加变量，删除变量
x = []
variate_add = []
variate_del = variate.copy()

Y = random.sample(variate, 3)  # 随机生成一个模型，3为变量的个数

# 将随机生成的三个变量分别输入到 添加变量和删除变量的数组
for i in Y:
    variate_add.append(i)
    x.append(i)
    variate_del.remove(i)
global aic  # 设置全局变量 这里选择AIC值作为指标
formula = "{}~{}".format("Y", "+".join(variate_add))  # 将自变量名连接起来
aic = smf.ols(formula=formula, data=df).fit().aic  # 获取随机函数的AIC值，与后面的进行对比
print("随机选择模型为：{}~{}，对应的AIC值为：{}".format("Y", "+".join(variate_add), aic))
print("剩余变量为：{}".format(variate_del))
print("\n")


# 添加变量
def forward():
    score_add = []
    global best_add_score
    global best_add_c
    print("添加变量")
    for c in variate_del:
        formula1 = "{}~{}".format("Y", "+".join(variate_add + [c]))
        score = smf.ols(formula=formula1, data=df).fit().aic
        score_add.append((score, c))  # 将添加的变量，以及新的AIC值一起存储在数组中

        print('自变量为{}，对应的AIC值为：{}'.format("+".join(variate_add + [c]), score))

    score_add.sort(reverse=True)  # 对数组内的数据进行排序，选择出AIC值最小的
    best_add_score, best_add_c = score_add.pop()

    print("最小AIC值为：{}".format(best_add_score))
    print("\n")


# 删除变量
def backward():
    score_del = []
    global best_del_score
    global best_del_c
    print("剔除变量")
    for i in x:
        select = x.copy()  # copy一个集合，避免重复修改到原集合
        select.remove(i)
        formula2 = "{}~{}".format("Y", "+".join(select))
        score = smf.ols(formula=formula2, data=df).fit().aic
        print('自变量为{}，对应的AIC值为：{}'.format("+".join(select), score))
        score_del.append((score, i))

    score_del.sort(reverse=True)  # 排序，方便将最小值输出
    best_del_score, best_del_c = score_del.pop()  # 将最小的AIC值以及对应剔除的变量分别赋值
    print("最小AIC值为：{}".format(best_del_score))
    print("\n")


forward()
backward()

while variate:

    if aic < best_add_score < best_del_score or aic < best_del_score < best_add_score:
        print("当前回归方程为最优回归方程，为{}~{}，AIC值为：{}".format("Y", "+".join(variate_add), aic))
        break
    elif best_add_score < best_del_score < aic or best_add_score < aic < best_del_score:
        print("目前最小的aic值为{}".format(best_add_score))
        print('选择自变量：{}'.format("+".join(variate_add + [best_add_c])))
        print('\n')
        variate_del.remove(best_add_c)
        variate_add.append(best_add_c)
        print("剩余变量为：{}".format(variate_del))
        aic = best_add_score
        forward()
    else:
        print('当前最小AIC值为：{}'.format(best_del_score))
        print('需要剔除的变量为：{}'.format(best_del_c))
        aic = best_del_score  # 将AIC值较小的选模型AIC值赋给aic再接着下一轮的对比
        x.remove(best_del_c)  # 在原集合上剔除选模型所对应剔除的变量
        backward()
