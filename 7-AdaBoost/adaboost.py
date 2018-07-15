# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: adaboost.py 
@time: 2018-04-25 20:59  
"""
import numpy as np
import matplotlib.pyplot as plt


def loadDataSet1():
    # 直接返回numpy矩阵
    dataMat = np.matrix([[1., 2.1],
                         [2., 1.1],
                         [1.3, 1.],
                         [1., 1.],
                         [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels


# 根据提供的参数来构造单层的决策树
# dimen —— 表示采用哪个维度，即采用哪个特征值进行单层的决策树划分
# threshaVal —— 表示决策树划分的阈值
# threshIneq —— 表示决策树划分的不等式的方向，即采用 <= 还是采用 >
def strumpClassify(dataMatrix, dimen, threshaVal, threshIneq):
    retArray = np.ones((dataMatrix.shape[0], 1))
    if threshIneq == 'lt':
        # 利用数组的过滤来将dataMatrix中第dimen列的值小于threshaVal的数据全部置为-1
        retArray[dataMatrix[:, dimen] <= threshaVal] = -1
    else:
        retArray[dataMatrix[:, dimen] > threshaVal] = -1
    return retArray


# 根据数据集的权重D来构造一颗错误率最小的单层决策树
def buildStump(dataArr, classLabel, D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabel).T
    m, n = dataMatrix.shape
    numSteps = 10
    bestStump = {}
    bestClassEst = np.zeros((m, 1))
    # float('inf') 表示正无穷, -float('inf') 或 float('-inf') 表示负无穷
    minError = float('inf')
    # 遍历每个特征
    for i in range(n):
        minValue = min(dataMatrix[:, i])
        maxValue = max(dataMatrix[:, i])
        stepSize = (maxValue - minValue) / numSteps
        # 遍历每个阈值点
        for j in range(-1, numSteps + 1):
            threshVal = float(minValue + stepSize * j)
            # 遍历每个不等式的方向
            for inequal in ['lt', 'gt']:
                predictVals = strumpClassify(dataMatrix, i, threshVal, inequal)
                errorArr = np.ones((m, 1))
                errorArr[predictVals == labelMat] = 0
                weightedError = float(np.dot(D.T, errorArr))
                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.2f"
                      % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst


# 使用adaBost进行训练，这里的弱分类器采用决策树桩
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = dataArr.shape[0]
    D = np.ones((m, 1)) / m
    aggClassEst = np.zeros((m, 1))
    for i in range(numIt):
        bestStump, minError, bestClassEst = buildStump(dataArr, classLabels, D)
        print('D.T = ', D.T)
        # 得到此分类器的权重
        alpha = float(0.5 * np.log((1.0 - minError) / max(minError, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print('bestClassEst:')
        print(bestClassEst)
        # 根据分类器的权重来更新数据集中每个样本的权重
        expon = -1 * alpha * np.multiply(np.mat(classLabel).T, bestClassEst)
        D = np.multiply(D, np.exp(expon))
        D = D / np.sum(D)
        # 将弱分类器根据其权重进行集成
        aggClassEst += alpha * bestClassEst
        print('aggClassEst:')
        print(aggClassEst)
        # 根据集成后的强分类器来得到分类结果
        aggError = np.multiply((np.sign(aggClassEst) != np.mat(classLabel).T), np.ones((m, 1)))
        errorRate = aggError.sum() / m
        print('total error =', errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst


# 根据训练好的分类器来对测试集进行测试
# datToClass —— 需要测试的测试数据集集
# classifierArr —— 训练好的集成的分类器
def adaClassify(datToClass, classifierArr):
    dataMatrix = np.mat(datToClass)
    m = dataMatrix.shape[0]
    aggClassEst = np.zeros((m, 1))
    for i in range(len(classifierArr)):
        classEst = strumpClassify(dataMatrix, classifierArr[i]['dim'],
                                  classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print('第{}次迭代得到的集成结果：'.format(i + 1), aggClassEst)
    return np.sign(aggClassEst)


# 从文件中装载数据集
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    for line in open(fileName).readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        labelMat.append(float(curLine[-1]))
        dataMat.append(lineArr)
    return np.matrix(dataMat), labelMat


def plotROC(predStrengths, classLabels):
    cur = (1.0, 1.0)
    ySum = 0.0
    numPosClas = sum(np.array(classLabel) == 1.0)
    yStep = 1 / numPosClas
    xStep = 1 / (len(classLabel) - numPosClas)
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        # 如果classLabel[index]为正例，则需要对真阳率进行修改
        if classLabel[index] == 1.0:
            delx = 0
            dely = yStep
        # 如果classLabel[index]为反例，则需要对假阳率进行修改
        else:
            delx = xStep
            dely = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0] - delx], [cur[1], cur[1] - dely], c='b')
        cur = (cur[0] - delx, cur[1] - dely)
    ax.plot([0, 1], [0, 1], 'b--')  # 在(0,0)和(1,1)这两个点之间画一条虚线
    ax.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    plt.show()
    print('The Area Under the Curve is:', ySum * xStep)


dataMat, classLabel = loadDataSet('horseColicTraining2.txt')
# m = dataMat.shape[0]
# D = np.ones((m, 1)) * (1 / m)
# print(buildStump(dataMat, classLabel, D))
# classifierArr = adaBoostTrainDS(dataMat, classLabel, numIt=50)
# print('-------------------------------------------------')
# testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
# prediction = adaClassify(testArr, classifierArr)
# errArr = np.ones((len(testArr), 1))
# # 使用numpy数组的过滤功能来计算预测错误的样本个数
# errorCount = np.sum(errArr[prediction != np.mat(testLabelArr).T])
# print('测试集的错误率为：%.2f' % (errorCount / prediction.shape[0]))
print('----------------------------------------------------')
classifierArr, aggClassEst = adaBoostTrainDS(dataMat, classLabel, numIt=10)
plotROC(aggClassEst.T, classLabel)
