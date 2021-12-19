'''
1.实例化，建立评估模型对象
2.通过模型接口训练模型
3.通过模型接口提取需要的信息
'''

import matplotlib
from sklearn import tree
from sklearn.datasets import load_wine
# 分测试集和训练集的类
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
# 画树
import graphviz
# 用于转换表格
import pandas as pd

# 实例化
'''
不纯度基于节点来计算，树中的每个节点都会有一个不纯度，并且子节点的不纯度一定低于父节点，
在同一颗决策树上，叶子节点的不纯度一定是最低的

criterion: 尺度，决定不纯度的计算方法
    gini: 默认参数，基尼系数

    entropy: 信息熵
        sklearn 实际计算的是基于信息熵的信息增益【information gain】，即父节点的信息熵和子节点的信息熵之差。
    
    比起基尼系数，信息熵对不纯度更加敏感，对不纯度的惩罚最强。但是在实际使用中，信息熵和基尼系数的效果基本相同。
    信息熵的计算速度比基尼系数缓慢一些，因为基尼系数的计算不涉及对数。
    另外，因为信息熵对不纯度更加敏感，所以信息熵作为指标时，决策树的生长会更加“精细”，因此对于高维数据或噪音很多的数据，
    信息熵很容易过拟合，基尼系数在这种情况下效果往往更好。当然，不是绝对的。
'''
wine = load_wine()

X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)

# print(X_train)
# (124, 13)
# print(X_train.shape)
# (54, 13)
print(X_test.shape)

# 1.实例化，创建一个分类器
# random_state, splitter 防止过拟合
# 剪枝
# min_samples_leaf 一个节点在分支后至少包含 min_simple_leaf 个训练样本，一般搭配 max_depth 使用，
# max_depth 如果太小会过拟合，太大会组织学习数据，一般从 5 开始
# min_samples_split 一个节点至少包含 min_samples_split 个训练样本才允许分支，否则分支不会发生
# max_features 限制分枝时考虑的特征个数，超过限制个数的特征都会被舍弃
# min_impurity_decrease 限制信息增益大小，信息增益小于设定值的分支不会发生。
test = []
for i in range(10):
    clf = tree.DecisionTreeClassifier(criterion = "entropy"
                                    ,random_state=65
                                    ,splitter="best"
                                    ,max_depth=i+1
                                    # ,min_samples_leaf=10
                                    # ,min_samples_split=10
                                    )

    # 用训练集数据训练模型
    clf = clf.fit(X_train, y_train)
    # 导入测试集，从接口中调用需要的信息，例如 score： 测试的准确度 accuracy
    result = clf.score(X_test, y_test)
    print("result: %5.16f" % result)
    # 输出对于测试集的预测结果
    # clf.predict(X_test)
    # 确定最优剪枝参数：
    #   超参数的曲线来进行判断，是一条以超参数的取值为横坐标，模型的度量指标为纵坐标的曲线。用来
    #   衡量不同超参数去之下模型的表现的线。
    test.append(result)
plt.plot(range(1,11), test, color="red", label = "max_depth")
plt.legend()
plt.show()

feature_name = [
    "酒精", "苹果酸", "灰", "灰的碱性",
    "镁", "总酚", "类黄酮", "非黄烷类分类",
    "花青素", "颜色强度", "色调", "od280/od315稀释葡萄酒",
    "脯氨酸"
]
# 画决策树
# filled - 填充颜色
# rounded - 图片的圆角
# graphviz
dot_data = tree.export_graphviz(
    decision_tree = clf
    ,feature_names = feature_name
    ,class_names = ["琴酒", "雪梨", "贝尔摩德"]
    ,filled = True
    ,rounded = True
)
graph = graphviz.Source(dot_data)
# jupeterlab:
# graph
# save on disk: 
graph.render(outfile='wine.png', format='png', cleanup = False)
'''
[[1.423e+01 1.710e+00 2.430e+00 ... 1.040e+00 3.920e+00 1.065e+03]
 [1.320e+01 1.780e+00 2.140e+00 ... 1.050e+00 3.400e+00 1.050e+03]
 [1.316e+01 2.360e+00 2.670e+00 ... 1.030e+00 3.170e+00 1.185e+03]
 ...
 [1.327e+01 4.280e+00 2.260e+00 ... 5.900e-01 1.560e+00 8.350e+02]
 [1.317e+01 2.590e+00 2.370e+00 ... 6.000e-01 1.620e+00 8.400e+02]
 [1.413e+01 4.100e+00 2.740e+00 ... 6.100e-01 1.600e+00 5.600e+02]]
'''
# print(wine.data)

'''
3分类的数据
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
'''
# print(wine.target)

# 数据结构：178行，13列。(178, 13个特征)
# print(wine.data.shape)

'''
        0     1     2     3      4     5     6     7     8      9     10    11      12  0 
0    14.23  1.71  2.43  15.6  127.0  2.80  3.06  0.28  2.29   5.64  1.04  3.92  1065.0   0
1    13.20  1.78  2.14  11.2  100.0  2.65  2.76  0.26  1.28   4.38  1.05  3.40  1050.0   0
2    13.16  2.36  2.67  18.6  101.0  2.80  3.24  0.30  2.81   5.68  1.03  3.17  1185.0   0
3    14.37  1.95  2.50  16.8  113.0  3.85  3.49  0.24  2.18   7.80  0.86  3.45  1480.0   0
4    13.24  2.59  2.87  21.0  118.0  2.80  2.69  0.39  1.82   4.32  1.04  2.93   735.0   0
..     ...   ...   ...   ...    ...   ...   ...   ...   ...    ...   ...   ...     ...  ..
173  13.71  5.65  2.45  20.5   95.0  1.68  0.61  0.52  1.06   7.70  0.64  1.74   740.0   2
174  13.40  3.91  2.48  23.0  102.0  1.80  0.75  0.43  1.41   7.30  0.70  1.56   750.0   2
175  13.27  4.28  2.26  20.0  120.0  1.59  0.69  0.43  1.35  10.20  0.59  1.56   835.0   2
176  13.17  2.59  2.37  20.0  120.0  1.65  0.68  0.53  1.46   9.30  0.60  1.62   840.0   2
177  14.13  4.10  2.74  24.5   96.0  2.05  0.76  0.56  1.35   9.20  0.61  1.60   560.0   2
'''
# print(
#     pd.concat([pd.DataFrame(wine.data), pd.DataFrame(wine.target)], axis = 1)
# )

'''
[
    'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
    'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
    'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'
]

'''
# print(wine.feature_names)

# ['class_0' 'class_1' 'class_2']
# print(wine.target_names)

