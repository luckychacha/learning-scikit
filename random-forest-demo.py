'''
集成学习是试下非常流行的机器学习算法，通过在数据上构建多个模型，集成所有模型的建模结果。
常见的有随机森林，梯度提升术【gbdt】，Xgboost 等。
集成算法的目标：
    集成算法考虑多个评估器的建模结果，汇总之后得到一个综合的结果，以此来获取比单个模型更好的
    回归或分类表现。
多个模型集成成为的模型叫做集成评估器，组成集成评估器的每个模型都叫做基评估器，通常来说，有三类
集成算法：装袋法【bagging】，提升法【boosting】 和 stacking。
    1. 装袋法的核心思想是构建多个相互独立的评估器，然后对其预测进行平均或多数表决原则来决定评估器的结果。
装袋法的代表就是随机森林。
    2. 提升法中，基评估器是相关的，是按顺序一一构建的。其核心思想是结合弱评估器的力量一次次对难以评估的样本
进行预测，从而构成一个强评估器。提升法的代表模型有 Adaboost 和梯度提升树。

n_estimators 森林中树木的数量。这个参数对于模型的精确度是单调的，这个值越大，模型的效果往往越来越好了。但相应的，任何模型都有决策边界，
到一定程度之后精确性往往不再上升或者开始波动。
'''

# %matplotlib inline
# 加上这句话的作用就是可以省略 plt.show 

from numpy.core.numeric import cross
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine

wine = load_wine()


# 3.使用其他接口将测试集导入我们训练好的模型，去获取我们希望获取的结果（score, Y_test）
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data, wine.target, test_size=0.3)

# 1.实例化
clf = DecisionTreeClassifier(random_state=0)
rfc = RandomForestClassifier(random_state=0)

# 2.训练集带入实例化后的模型进行训练，使用的接口是 fit
clf = clf.fit(Xtrain, Ytrain)
rfc = rfc.fit(Xtrain, Ytrain)

score_c = clf.score(Xtest, Ytest)
score_r = rfc.score(Xtest, Ytest)
print("Single Tree: {}".format(score_c), "Random Forest: {}".format(score_r))

# 交叉验证: cross_val_score

from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


rfc = RandomForestClassifier(n_estimators=25)
# # cv 交叉验证的次数。
rfc_s = cross_val_score(rfc, wine.data, wine.target, cv=10)

clf = DecisionTreeClassifier()
clf_s = cross_val_score(clf, wine.data, wine.target, cv=10)

plt.plot(range(1,11), rfc_s, label="Random Forest")
plt.plot(range(1,11), clf_s, label="DecisionTree")
# # 显示图例-标签
plt.legend()
plt.show()

'''
label = "RandomForest"
for model in [RandomForestClassifier(n_estimators=25), DecisionTreeClassifier()]:
    score = cross_val_score(model, wine.data, wine.target, cv=10)
    print("{}:".format("label"), print(score.mean()))
    plt.plot(range(1, 11), score, label = label)
    label = "DecisionTree"

plt.legend()
plt.show()
'''
# 画出随机森林和决策树在 10 组交叉验证下的效果对比。【通常不用做】
'''
rfc_l = []
clf_l = []
for i in range(10):
    rfc = RandomForestClassifier(n_estimators=25)
    rfc_s = cross_val_score(rfc, wine.data, wine.target).mean()
    rfc_l.append(rfc_s)
    
    clf = DecisionTreeClassifier()
    clf_s = cross_val_score(clf, wine.data, wine.target).mean()
    clf_l.append(clf_s)

plt.plot(range(1, 11), rfc_l, label = "Random Forest")
plt.plot(range(1, 11), clf_l, label = "Decision Tree")
plt.legend()
plt.show()
'''

# n_estimators 的学习曲线
superpa = []
for i in range(200):
    rfc = RandomForestClassifier(n_estimators=i+1, n_jobs=-1)
    rfc_s = cross_val_score(rfc, wine.data, wine.target, cv=10).mean()
    superpa.append(rfc_s)

# 0.9888888888888889 28
# 表示第 28 组准确率最高，29 棵树的森林准确率最高
print(max(superpa), superpa.index(max(superpa)))

plt.figure(figsize=[20, 5])
plt.plot(range(1, 201), superpa)
plt.show()