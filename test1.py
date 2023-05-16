# Author: 史浩 浙江金融职业学院
# --------------------------------------------------------
import numpy as np
import pandas as pd

nsample = 20
D1 = pd.DataFrame(data={"city": [1 for i in range(1, 10)]})  # 农村居民=1
D0 = pd.DataFrame(data={"city": [0 for i in range(1, 10)]})  # 城市居民=0
D = pd.concat([D1, D0]).reset_index(drop=True)
print(D1)
print(D0)
print(D)
