# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: trees.py 
@time: 2018-04-20 9:34  
"""
import numpy as np
import operator

try:
    from treePlotter import createPlot
except ImportError:
    raise ImportError('The file is not found. Please check the file name!')


# 计算数据集的信息熵
def calcShannonEnt(dataSet):
    m = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 1
        else:
            labelCounts[currentLabel] += 1
    shannonEnt = 0.0  # 计算信息熵
    for key in labelCounts.keys():
        prob = float(labelCounts[key]) / m
        shannonEnt -= prob * np.log2(prob)  # np.log2()函数表示以2为底的log函数
    return shannonEnt


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    # labels表示每个特征表示的意思
    labels = ['不浮出水面是否可以生存', '是否有脚蹼']
    return dataSet, labels


# 按照给定特征的给定值划分数据集，返回特征axis=value的样本的集合
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 需要将axis所表示的特征从数据集中去除(因为该特征已经使用)
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])  # 在这里需要使用extend
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 根据信息增益来选择最好的特征
def chooseBestFeatureToSplit(dataSet):
    m = float(len(dataSet))
    # 获取特征个数,dataSet是list，并且其中每个元素都是list
    numFeatures = len(dataSet[0]) - 1
    bestFeature = -1
    bestInfoGain = 0.0
    baseEntropy = calcShannonEnt(dataSet)
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueValues = set(featList)
        newEntropy = 0.0
        for value in uniqueValues:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / m
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestFeature = i
            bestInfoGain = infoGain
    return bestFeature


# 当属性用完时，但是类标签依然不是唯一的，此时我们需要决定如何定义该叶子节点
# 这里使用多数投票法
def majorityCnt(classList):
    classLabel = {}
    for vote in classList:
        if vote not in classLabel.keys():
            classLabel[vote] = 0
        classLabel[vote] += 1
    sortedClassCount = sorted(classLabel.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 使用递归的方式创建决策树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 递归函数的停止条件1：所有的类标签完全相同，则直接返回类标签
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 递归函数的停止条件2：使用完了所有的特征，仍然不能将数据集划分成包含唯一类别的分组
    if len(dataSet[0]) == 1:
        # 使用多数投票法来决定该叶子节点的分类
        return majorityCnt(classList)
    # 选取信息增益最大的特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    # MyTree用于存储树的所有信息
    MyTree = {bestFeatLabel: {}}
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLables = labels[:bestFeat]
        subLables.extend(labels[bestFeat + 1:])
        MyTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,
                                                               bestFeat, value), subLables)
    return MyTree


# myTree的格式为：{'不浮出水面是否可以生存': {0: 'no', 1: {'是否有脚蹼': {0: 'no', 1: 'yes'}}}}
def classify(inputTree, featLabels, testVec):
    classLabel = ""
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    # index()方法用于查找当前列表中第一个匹配firstStr变量的元素
    featIndex = featLabels.index(firstStr)  # featIndex表示根节点所在的索引
    for key in secondDict.keys():
        if testVec[featIndex] == key:  # 找到testVec的featIndex属性对应的值
            if type(secondDict[key]).__name__ == "dict":
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


# 使用pickle模块存储决策树
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, "wb")
    pickle.dump(inputTree, fw)
    fw.close()


# 将存储到硬盘中的决策树读取出来
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


# myDat, labels = createDataSet()
# print("数据集为：", myDat)
# print("此数据集的信息熵为：", calcShannonEnt(myDat))
# print(splitDataSet(myDat, 0, 1))
# print(splitDataSet(myDat, 0, 0))
# print("信息增益最大的特征为：", labels[chooseBestFeatureToSplit(myDat)])
# myTree = createTree(myDat, labels)
# print("生成的树为：", myTree)
# createPlot(myTree)
# print('[1, 0]对应的标签为', classify(myTree, labels, [1, 0]))
# storeTree(myTree, "classifierStorage.txt")

fr = open("lenses.txt")
lenses = [line.strip().split("\t") for line in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
myTree = createTree(lenses, lensesLabels)
print("生成的树为：", myTree)
createPlot(myTree)
# print('[1, 0]对应的标签为', classify(myTree, lensesLabels, [1, 0]))
storeTree(myTree, "lensesStorage.txt")
