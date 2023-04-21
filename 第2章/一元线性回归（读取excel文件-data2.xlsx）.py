import pandas as pd
import statsmodels.api as sm

df = pd.read_excel(r'data2.xlsx')
df = df.iloc[:, :5]
# print(df)
#              0      1          2      3       4
df.columns = ['学校', 'ASP/美元', 'GPA', 'GMAT', '学费/美元']
df.reset_index(drop=True, inplace=True)  # 把索引重新排一下
# print(df)

# 定义变量
# 用.astype(float)将EXCEL文件中的字符型转浮点型
asp = df.iloc[:, 1].astype(float)
gpa = df.iloc[:, 2].astype(float)
gmat = df.iloc[:, 3].astype(float)
tuition = df.iloc[:, 4].astype(float)

##########################################################################
# 进行一元线性回归，可以替换Y和X
#          OLS：  Y          = beta0 + beta1*X
##########################################################################
model = sm.OLS(asp / 10000, sm.add_constant(gpa))  # 用add_constant加入常数项
print(model.fit().summary())
