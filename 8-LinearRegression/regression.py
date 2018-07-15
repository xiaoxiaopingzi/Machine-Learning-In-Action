# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: regression.py 
@time: 2018-04-26 19:42

线性回归
"""
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import json
from urllib.request import urlopen


# 从文件中装载数据集，该数据集的第一个特征为1
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
    return dataMat, labelMat


# 使用法方程的方法来得到线性回归的最佳拟合参数
def standRegres(xArr, yArr):
    # xMat的第一个特征为1(即已经加入了常数项)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = np.dot(xMat.T, xMat)
    # np.linalg.det(xTx)表示对矩阵求行列式
    # np.linalg中一个线性代数的库，包括矩阵求逆，矩阵求行列式等多多种操作
    # 判断矩阵的行列式是否为0，因为如果矩阵的行列式为0，则矩阵式不可逆的
    if np.linalg.det(xTx) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    # 矩阵求逆：f = np.linalg.inv(a)  or  f = a ** (-1)  or  f = a.I
    ws = np.dot(np.dot(np.linalg.inv(xTx), xMat.T), yMat)
    return ws


# 局部加权线性回归
# 局部加权线性回归增加了计算量，因为它对每个点进行预测时都必须使用整个数据集
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = xMat.shape[0]
    # 初始的权重矩阵为单位矩阵
    weights = np.eye((m))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(np.dot(diffMat, diffMat.T) / (-2 * k * k))
    xTx = np.dot(np.dot(xMat.T, weights), xMat)
    if np.linalg.det(xTx) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = np.dot(np.dot(np.dot(np.linalg.inv(xTx), xMat.T), weights), yMat)
    return np.dot(testPoint, ws)


# 测试局部加权线性回归
def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = testArr.shape[0]
    yHat = np.zeros((m, 1))
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


# 根据yArr和yHatArr的值来计算真实值和预测值之间的均方误差
def rssError(yArr, yHatArr):
    return np.sum(np.power(yArr - yHatArr, 2))


# 使用岭回归来拟合参数
def ridgeRegres(xMat, yMat, lam=0.2):
    denom = np.dot(xMat.T, xMat) + lam * np.eye((xMat.shape[1]))
    if np.linalg.det(denom) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = np.dot(np.dot(np.linalg.inv(denom), xMat.T), yMat)
    return ws


# 使用不同的lambda来测试岭回归的拟合效果
def ridgeTest(xArr, yArr):
    xMat, yMat = regularize(xArr, yArr)
    numTestPts = 30
    wMat = np.zeros((numTestPts, xMat.shape[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, np.exp(i - 10))
        wMat[i, :] = ws.T
    return wMat


# 使用前向逐步回归来拟合系数，采用了贪心算法的思想
def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat, yMat = regularize(xArr, yArr)
    m, n = xMat.shape
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        lowestError = float('inf')
        print(ws.T)
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += sign * eps
                yTest = np.dot(xMat, wsTest)
                rssE = rssError(yMat, yTest)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat


# 对输入的数据进行归一化
def regularize(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)  # 列的均值
    xMean = np.mean(xMat, 0)  # 列的均值
    yMat = yMat - yMean
    xMat = xMat - xMean
    xVar = np.var(xMat, 0)
    xMat = xMat / xVar
    return xMat, yMat


# 测试标准的线性回归
def testLR():
    dataMat, labelMat = loadDataSet('ex0.txt')
    ws = standRegres(dataMat, labelMat)
    dataMatrix = np.mat(dataMat)
    yHat = np.dot(dataMatrix, ws)
    x = [i[1] for i in dataMat]
    fig = plt.figure()
    plt.scatter(x, labelMat, s=20, c='r')
    plt.plot(x, yHat)
    plt.show()
    # 计算预测值和真实值之间的相关性，注意这里的向量必须是行向量
    print(np.corrcoef(yHat.T, np.mat(labelMat)))


# 测试局部加权线性回归
def testlwlr():
    dataMat, labelMat = loadDataSet('ex0.txt')
    dataMatrix = np.mat(dataMat)
    # k的值过小会导致过拟合
    yHat = lwlrTest(dataMatrix, dataMat, labelMat, 1)
    yHat2 = lwlrTest(dataMatrix, dataMat, labelMat, 0.01)
    yHat3 = lwlrTest(dataMatrix, dataMat, labelMat, 0.003)
    x = [i[1] for i in dataMat]
    srtInd = np.argsort(x)
    xNew = []
    yHatNew = []
    yHatNew2 = []
    yHatNew3 = []
    for i in srtInd:
        xNew.append(x[i])
        yHatNew.append(yHat[i])
        yHatNew2.append(yHat2[i])
        yHatNew3.append(yHat3[i])
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)
    ax1.scatter(x, labelMat, s=15, c='r')
    ax1.plot(xNew, yHatNew)
    ax2.scatter(x, labelMat, s=15, c='r')
    ax2.plot(xNew, yHatNew2)
    ax3.scatter(x, labelMat, s=15, c='r')
    ax3.plot(xNew, yHatNew3)
    plt.show()


# 测试局部加权线性回归
# dataMat, labelMat = loadDataSet('abalone.txt')
# dataMatrix = np.mat(dataMat)
# yHat = lwlrTest(dataMatrix[100:199], dataMat[0:99], labelMat[0:99], 1)
# print(rssError(np.mat(labelMat[100:199]).T, yHat))

# 测试岭回归
# dataMat, labelMat = loadDataSet('abalone.txt')
# riderWeights = ridgeTest(dataMat, labelMat)
# fig = plt.figure()
# plt.plot(riderWeights)
# plt.show()

# 测试前向逐步回归
# dataMat, labelMat = loadDataSet('abalone.txt')
# stageWiseWeights = stageWise(dataMat, labelMat, 0.001, 5000)
# print('---------------------标准的线性回归-----------------------------')
# xMat, yMat = regularize(dataMat, labelMat)
# fig = plt.figure()
# ws = standRegres(xMat, yMat.T)
# print(ws)
# plt.plot(stageWiseWeights)
# plt.show()


def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (
        myAPIstr, setNum)
    pg = urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else:
                newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if sellingPrice > origPrc * 0.5:
                    print
                    "%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice)
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except:
            print('problem with item %d' % i)


def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)

# 使用交叉验证测试岭回归
def crossValidation(xArr, yArr, numVal=10):
    m = len(yArr)
    indexList = range(m)
    errorMat = zeros((numVal, 30))  # create error mat 30columns numVal rows
    for i in range(numVal):
        trainX = []
        trainY = []
        testX = []
        testY = []
        # 随机打乱数据集
        random.shuffle(indexList)
        for j in range(m):  # create training set based on first 90% of values in indexList
            if j < m * 0.9:  # 90%的数据作为训练集
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX, trainY)  # get 30 weight vectors from ridge
        for k in range(30):  # loop over all of the ridge estimates
            matTestX = mat(testX);
            matTrainX = mat(trainX)
            meanTrain = mean(matTrainX, 0)
            varTrain = var(matTrainX, 0)
            matTestX = (matTestX - meanTrain) / varTrain  # regularize test with training params
            yEst = matTestX * mat(wMat[k, :]).T + mean(trainY)  # test ridge results and store
            errorMat[i, k] = rssError(yEst.T.A, array(testY))
            # print errorMat[i,k]
    meanErrors = mean(errorMat, 0)  # calc avg performance of the different ridge weight vectors
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors == minMean)]
    # can unregularize to get model
    # when we regularized we wrote Xreg = (x-meanX)/var(x)
    # we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
    xMat = mat(xArr);
    yMat = mat(yArr).T
    meanX = mean(xMat, 0);
    varX = var(xMat, 0)
    unReg = bestWeights / varX
    print("the best model from Ridge Regression is:\n", unReg)
    print("with constant term: ", -1 * sum(multiply(meanX, unReg)) + mean(yMat))
