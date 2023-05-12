# coding=utf-8
# Author: 史浩 浙江金融职业学院
# https://blog.csdn.net/DL11007/article/details/129196843
# --------------------------------------------------------
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

from itertools import combinations


# 求所有可能的变量组合
def comb(a):  # 参数a为所有自变量的列表
    all = []
    for i in range(len(a) + 1):
        num_choose = i  # 从列表a中选num_choose个变量
        combins = [c for c in combinations(a, num_choose)]
        for i in combins:
            i = list(i)
            all.append(i)
    return all


# 对列表variables中的变量进行组合并建立模型,参数df为读入的数据
def buildModel(variables, df):
    combine = ''
    for variable in variables:
        combine = combine + variable + '+'  # 对列表中的变量进行组合
    combine = combine[:-1]
    if len(combine) == 0:
        combine = '1'

    result = smf.ols('y~' + combine, data=df).fit()  # 得出回归结果
    return result


# 变量列表和对应的aic以dataframe的形式输出
def printDataFrame(model, aic):  # model为所有备选变量组合的列表，aic为所有备选模型aic的列表
    data = {'model': model, 'aic': aic}  # 输出Dataframe形式
    frame = pd.DataFrame(data)
    print(frame)


# -----------------------------------------------------
# 所有子集回归（主要找AIC最小的自变量组合模型）
# -----------------------------------------------------
def allSubset(a, df):  # 参数a为所有自变量的列表，参数df为读入的数据
    all = []
    all = comb(a)  # 先获取所有变量组合组合
    aic = []  # 保存所有aic值
    for i in all:  # 遍历所有组合
        result = buildModel(i, df)  # 对每一种组合建立模型
        aic.append(result.aic)  # 获取aic并加入aic列表
    printDataFrame(all, aic)

    print("最小AIC为：{}".format(min(aic)))
    index = aic.index(min(aic))
    print("里面包含的自变量为：{}".format(all[index]))
    temp = all[index]  # 获取最终变量组合

    result = buildModel(temp, df)  # 输出最终的回归结果
    print("回归结果：")
    print(result.summary())
    return temp


# --------------------------------------------------------------------
# 读入数据
df = pd.read_excel(r'..\..\data\美国1971-1986年期间新客车年销售量.xlsx', sheet_name="Sheet1")
df = df.iloc[:, :7]
df = df.dropna()
# print(df)
#              0      1    2     3     4     5     6
df.columns = ['year', 'y', 'x1', 'x2', 'x3', 'x4', 'x5']
df.reset_index(drop=True, inplace=True)  # 把索引重新排一下
# print(df)

# 定义变量
# 用.astype(float)将EXCEL文件中的字符型转浮点型
y = df.iloc[:, 1].astype(float).apply(np.log)
x1 = df.iloc[:, 2].astype(float).apply(np.log)
x2 = df.iloc[:, 3].astype(float).apply(np.log)
x3 = df.iloc[:, 4].astype(float).apply(np.log)
x4 = df.iloc[:, 5].astype(float).apply(np.log)
x5 = df.iloc[:, 6].astype(float).apply(np.log)

Z = np.column_stack((x1, x2, x3, x4, x5))
##########################################################################
# 进行多元线性回归，可以替换Y和X
#          OLS：  Y          = beta0 + beta1*X1 + beta2*X2
##########################################################################
model = sm.OLS(y, sm.add_constant(Z))  # 用add_constant加入常数项
print(model.fit().summary(xname=['const', 'lnx1', 'lnx2', 'lnx3', 'lnx4', 'lnx5']))  # 用自己的名称命名常数和各个解释变量

# 相关系数
print(np.corrcoef((x1, x2, x3, x4, x5)))


# --------------------------------------------------
# 方法一：所有子集回归
# 选择变量组合
a = ['x1', 'x2', 'x3', 'x4', 'x5']
x_df = pd.DataFrame(Z)  # 得到不含y的dataframe
print("-----------所有子集回归------------")
xvariables = allSubset(a, x_df)
print(xvariables)
# 几个变量就几行
# 求相关系数矩阵
array1 = np.zeros([len(xvariables), len(df)], dtype=np.float64)
for i, item in enumerate(xvariables):
    array1[i, :] = eval(item)

print(np.corrcoef(array1))

print("会发现还是有多重共线性，去掉一个x3")
a = ['x1', 'x2', 'x4']
xvariables = allSubset(a, x_df)
print(xvariables)
array1 = np.zeros([len(xvariables), len(df)], dtype=np.float64)
for i, item in enumerate(xvariables):
    array1[i, :] = eval(item)
print(np.corrcoef(array1))
print("现在会好一些了Cond. No.=753.")



#
# # 仅打印系数
# res = model.fit()
# print(res.params)
#
# df_out = pd.concat((res.params, res.tvalues, res.pvalues, res.conf_int()), axis=1)
# df_out.columns = ['beta', 't', 'p-value', 'CI_low', 'CI_high']
# print(df_out)
# # 输出到文件
# from openpyxl import Workbook
# from openpyxl.utils.dataframe import dataframe_to_rows
#
# wb = Workbook()
# ws = wb.create_sheet("sheet1")
# rows = dataframe_to_rows(df_out)
#
# for r_idx, row in enumerate(rows, 1):
#     for c_idx, value in enumerate(row, 1):
#         ws.cell(row=r_idx, column=c_idx, value=value)
# wb.save("output.xls")  # 输出到文件
