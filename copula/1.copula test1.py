# Author: 史浩 浙江金融职业学院
# --------------------------------------------------------
import matplotlib.pyplot as plt


from copulas.datasets import sample_trivariate_xyz
data = sample_trivariate_xyz()
# 原始三维数据
from copulas.visualization import scatter_3d
scatter_3d(data)
plt.show()

# 拟合
from copulas.multivariate import GaussianMultivariate
copula = GaussianMultivariate()
copula.fit(data)

# 比较
from copulas.visualization import compare_3d
num_samples = 1000
synthetic_data = copula.sample(num_samples)
compare_3d(data, synthetic_data)
plt.show()

# 保存和提取
# 对于需要较长时间进行拟合copula模型的数据，可以拟合一个比较合适的模型后，
# 用save保存这个模型，在每次想采样新数据时用load加载存储在磁盘上已经拟合好的模型。
# model_path = 'mymodel.pkl'
# copula.save(model_path)
# new_copula = GaussianMultivariate.load(model_path)
# new_samples = new_copula.sample(num_samples)
# 在某些情况下，从拟合的连接中获取参数比从磁盘中保存和加载参数更有用。可以使用to_dict方法提取copula模型的参数：
# copula_params = copula.to_dict()
# # 一旦有了所有的参数，就可以使用from_dict创建一个新的相同的Copula模型
# new_copula = GaussianMultivariate.from_dict(copula_params)
# # 用新模型生成新的参数：
# new_samples = new_copula.sample(num_samples)
