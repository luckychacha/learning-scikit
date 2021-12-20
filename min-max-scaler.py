from sklearn.preprocessing import MinMaxScaler

data = [[-1, 2], [-0.5, 6], [0, 10], [1, 120]]

import pandas as pd
pd.DataFrame(data)

# 实例化
scaler = MinMaxScaler()
# fit，在这里本质是生成 min(x) 和 max(x)，如果 fit 报错了，使用 partial_fit 训练接口，
scaler = scaler.fit(data)
# 通过接口导出结果
result = scaler.transform(data)

# print(result)

# 训练和导出一步完成
result_ = scaler.fit_transform(data)
# 将归一化的结果逆转，归一化之前的数据
reverse = scaler.inverse_transform(result)
# print(reverse)
# 使用 MinMaxScaler 的参数 feature_range 实现将数据归一化到 [0, 1] 以外的范围中
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
scaler = MinMaxScaler(feature_range=[5, 10])
result = scaler.fit_transform(data)
# print(result)

# 使用 numpy 来实现归一化
import numpy as np

X = np.array([[1,2], [-0.5, 6], [0, 10], [1, 18]])
# -0.5
print(X.min())
# [-0.5, 2]
print(X.min(axis=0))
# [ 1.  -0.5  0.   1. ]
print(X.min(axis=1))
x_nor = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
print(x_nor)