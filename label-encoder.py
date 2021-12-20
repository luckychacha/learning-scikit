from pandas.core.frame import DataFrame
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import numpy as np
import pandas as pd

data = pd.read_csv(r'./a.csv', index_col=0)
data_ = data.copy()
y = data.iloc[:, -1]
le = LabelEncoder()
le = le.fit(y)
lable = le.transform(y)
# ['No' 'Unknown' 'Yes']
print(le.classes_)
# [0 2 2 1]
print(lable)

data.iloc[:, -1] = LabelEncoder().fit_transform(data.iloc[:, -1])

#     Age     Sex Embarked  Survived
# id                                
# 0    22    male        S         0
# 1    23  female        C         2
# 2    66  female        S         2
# 3    77    male        S         1
print(data)

# 把还是字母的第二列和第三列也换为数字： [array(['female', 'male'], dtype=object), array(['C', 'S'], dtype=object)]
# print(OrdinalEncoder().fit(data_.iloc[:, 1:-1]).categories_)

data_.iloc[:, 1:-1] = OrdinalEncoder().fit_transform(data_.iloc[:, 1:-1])

#     Age  Sex  Embarked  Survived
# id                              
# 0    22  1.0       1.0         0
# 1    23  0.0       0.0         2
# 2    66  0.0       1.0         2
# 3    77  1.0       1.0         1
print(data_)
# 但是还不够好，例如舱门，应该使用哑变量，这样的变化能让算法彻底领悟三个值是没有可计算性的，是有你没我的不等概念。
# 也就是独热编码。
from sklearn.preprocessing import OneHotEncoder
# a = OneHotEncoder(categories='auto').fit_transform(data.iloc[:, 1:-1])
enc = OneHotEncoder(categories='auto').fit(data.iloc[:, 1:-1])
# ['x0_female' 'x0_male' 'x1_C' 'x1_S']
print(enc.get_feature_names())
result = enc.transform(data.iloc[:, 1:-1]).toarray()
#  性别分 2 中， Embarked 也分 2 种，所以一共 4 列数据。
# [[0. 1. 0. 1.]
#  [1. 0. 1. 0.]
#  [1. 0. 0. 1.]
#  [0. 1. 0. 1.]]
print(result)
#         0  1
# 0    male  S
# 1  female  C
# 2  female  S
# 3    male  S
# inverse = pd.DataFrame(enc.inverse_transform(result))
# print(inverse)
newdata = pd.concat([data, pd.DataFrame(result)], axis=1)
newdata.drop(['Sex', 'Embarked'], axis=1,inplace=True)

print(newdata)
newdata.columns = ["Age", "Survived", "x0_female", "x0_male", "x1_C", "x1_S"]
print(newdata)