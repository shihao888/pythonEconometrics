# Author: 史浩 浙江金融职业学院
#
# conda install pandas-datareader
# -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
#
# --------------------------------------------------------
import pandas_datareader as pdr
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
data1 = pdr.get_data_fred(['INDPRO', 'FEDFUNDS', 'UNRATE'])
formula = 'INDPRO~FEDFUNDS+UNRATE'
reg1 = smf.ols(formula, data1).fit()
print(reg1.summary())
uhat = reg1.resid
plt.plot(uhat)
plt.xlabel('Date')
plt.ylabel('Residual')
plt.grid(True)
plt.show()