# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: regTrees.py 
@time: 2018-04-28 16:03  
"""
import numpy as np
import matplotlib.pyplot as plt


# 从文件中装载数据
def loadDataSet(fileName):  # general function to parse tab -delimited floats
    dataMat = []  # assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = [float(i) for i in curLine]
        dataMat.append(fltLine)
    return dataMat


# 按照指定的切分方式对数据集进行切分
def binSplitDataSet(dataSet, feature, value):
    # np.nonzero用于返回非零元素的索引，其返回的形式为tuple
    # 采用数组过滤的方式切分数据
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


# 对数据集执行简单的线性回归得到回归系数
def linearSolve(dataSet):
    m, n = dataSet.shape
    X = np.ones((m, n))
    Y = np.zeros((m, 1))
    X[:, 1: n] = dataSet[:, 0: n - 1]
    Y = dataSet[:, -1]
    xTx = np.dot(X.T, X)
    if np.linalg.det(xTx) == 0:
        raise NameError('矩阵的行列式为0,矩阵不可逆')
    ws = np.dot(np.dot(np.linalg.inv(xTx), X.T), Y)
    return ws, X, Y


# 模型树的叶子节点的生成方式
def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws


# 模型树的叶子节点的误差
def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = np.dot(X, ws)
    return np.sum(np.square(yHat - Y))


# 回归树的叶子节点的生成方式，直接取数据集中数据的真实值得平均值
def regLeaf(dataSet):
    # 返回叶子节点的回归值
    return np.mean(dataSet[:, -1])


# 返回数据集的平方误差
def regErr(dataSet):
    m = dataSet.shape[0]
    return np.var(dataSet[:, -1]) * m


# 选择最好的切分方式
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    tols = ops[0]
    tolN = ops[1]
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = dataSet.shape
    S = errType(dataSet)
    bestFeature = 0
    bestValue = 0
    bestS = float('inf')
    for featIndex in range(n - 1):
        for splitVal in set(dataSet[:, featIndex].tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            # 如果切分后得到的数据集的太小，则直接下一次循环
            if (mat0.shape[0] < tolN) or (mat1.shape[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestS = newS
                bestFeature = featIndex
                bestValue = splitVal
    # 如果切分后减少的误差过小或者切分后得到的数据集过小，则不进行切分(这是一种预剪枝的策略)
    if (S - bestS) < tols:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestFeature, bestValue)
    if (mat0.shape[0] < tolN) or (mat1.shape[0] < tolN):
        return None, leafType(dataSet)
    return bestFeature, bestValue


# 创造回归树
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)  # choose the best split
    if feat is None:
        return val  # if the splitting hit a stop condition return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


# 判断传进来的对象是否是树
def isTree(obj):
    return type(obj).__name__ == 'dict'


# 使用递归来获取树的均值
def getMean(tree):
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    return (tree['left'] + tree['right']) / 2


# 对生成好的树进行后剪枝，剪枝的策略是根据测试的误差的大小来决定是否进行剪枝
def prune(tree, testData):
    if testData.shape[0] == 0:
        return getMean(tree)
    if (isTree(tree['left'])) or (isTree(tree['right'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        if isTree(tree['left']):
            tree['left'] = prune(tree['left'], lSet)
        if isTree(tree['right']):
            tree['right'] = prune(tree['right'], rSet)
        return tree
    else:
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = np.sum(np.square(lSet[:, -1] - tree['left'])) \
                       + np.sum(np.square(rSet[:, -1] - tree['right']))
        treeMean = (tree['left'] + tree['right']) / 2
        errorMerge = np.sum(np.square(testData[:, -1] - treeMean))
        # 如果剪枝后得到的测试集误差小于未剪枝前的测试集误差，就说明可以进行剪枝
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree


# 回归树的叶子节点的值
def regTreeEval(model, inDat):
    return float(model)


# 模型树的叶子节点的值
def modelTreeEval(model, inDat):
    n = inDat.shape[1]
    X = np.ones((1, n + 1))
    X[0, 1: n + 1] = inDat
    returnValue = np.dot(X, model)
    return returnValue


# 根据建立好的回归树对输入的数据进行预测
def treeForceCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForceCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForceCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


def createForceCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    YHat = np.zeros((m, 1))
    for i in range(m):
        YHat[i, 0] = treeForceCast(tree, np.mat(testData[i]), modelEval)
    return YHat


# testMat = np.eye((4))
# print(testMat)
# print(testMat[:, 1:3])  # 左闭右开
# mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
# print(mat0)
# print('-----------------------------')
# print(mat1)
# print(help(np.nonzero))
# myDat = loadDataSet('ex00.txt')
# myDat = loadDataSet('ex2.txt')
# myMat = np.mat(myDat)
# # print(myMat)
# myTree = createTree(myMat, ops=(0, 1))
# print('剪枝前的树为：', myTree)
# testMat = np.mat(loadDataSet('ex2test.txt'))
# myTree2 = prune(myTree, testMat)
# print('剪枝后的树为：', myTree2)

# myMat = np.mat(loadDataSet('exp2.txt'))
# plt.scatter(myMat[:, 0].tolist(), myMat[:, 1].tolist(), c='r')
# myTree = createTree(myMat, modelLeaf, modelErr, ops=(1, 10))
# print(myTree)
# plt.show()


if __name__ == '__main__':
    # 判断回归树、模型树以及线性回归的优劣，通过相关系数来比较三者的优劣
    myTrainMat = np.mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    myTestMat = np.mat(loadDataSet('bikeSpeedVsIq_test.txt'))
    print('---------------------回归树：-----------------------------')
    myTree = createTree(myTrainMat, ops=(1, 20))
    print(myTree)
    yHat = createForceCast(myTree, myTestMat[:, 0])
    print(np.corrcoef(yHat, myTestMat[:, 1], rowvar=0))

    print('--------------------模型树：------------------------------')
    myTree = createTree(myTrainMat, modelLeaf, modelErr, ops=(1, 20))
    print(myTree)
    yHat = createForceCast(myTree, myTestMat[:, 0], modelTreeEval)
    print(np.corrcoef(yHat, myTestMat[:, 1], rowvar=0))

    print('---------------------标准的线性回归：-----------------------------------')
    ws, X, Y = linearSolve(myTrainMat)
    Yhat = np.zeros((myTestMat.shape[0], 1))
    for i in range(myTestMat.shape[0]):
        Yhat[i, 0] = ws[0, 0] + ws[1, 0] * myTestMat[i, 0]
    print(np.corrcoef(Yhat, myTestMat[:, 1], rowvar=0))
