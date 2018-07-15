# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: KNN.py 
@time: 2018-04-19 16:03  
"""
import numpy as np
import operator
from os import listdir


# 创建测试的数据集
def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ["A", "A", "B", "B"]
    return group, labels


# 根据KNN算法预测输入数据inx的类别
def classify0(inx, dataSet, labels, k):
    diffMat = np.square(dataSet - inx)
    distances = np.sqrt(np.sum(diffMat, axis=1))
    # 获取排序好的从小到大的距离的索引
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteLable = labels[sortedDistIndicies[i]]
        # classCount.get(voteLable, 0)表示取classCount字典中键为voteLable的值，
        # 如果classCount字典中没有对应的键值，则返回0
        classCount[voteLable] = classCount.get(voteLable, 0) + 1
    # d={"ok":1,"no":2}  #对字典按键排序，用元组列表的形式返回
    # [('ok', 1), ('no', 2)]
    # sortedClassCount = sorted(classCount.items(), key=lambda d: d[0], reverse=True)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 从文件中获取数据集
def file2mattix(filename):
    fr = open(filename)
    arrayOlines = fr.readlines()
    numberofLine = len(arrayOlines)
    # returnMat是需要返回的特征矩阵
    returnMat = np.zeros((numberofLine, 3))
    # classLabelVector存储每个训练样本的标签
    classLabelVector = []
    index = 0
    for line in arrayOlines:
        line = line.strip()
        listFromLine = line.split("\t")
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


# 将所有特征归一化, newValue = (oldValue - minValue) / (maxValue - minValue)
def autoNorm(dataSet):
    minValue = np.min(dataSet, axis=0)
    maxValue = np.max(dataSet, axis=0)
    range = maxValue - minValue
    normDataSet = (dataSet - minValue) / range
    return normDataSet, range, minValue


# 作为完整程序验证KNN分类器
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2mattix("datingTestSet2.txt")
    normMat, ranges, minValue = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)  # 选取前hoRatio的数据作为测试集，剩下的作为训练集
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:, :],
                                     datingLabels[numTestVecs:], 4)
        print("KNN分类器的分类结果为：{}，真实的标签为：{}".format(classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("分类器总的错误率为：{}%".format((errorCount / float(numTestVecs)) * 100))


# 作为一个单独的程序运行
def classifyPerson():
    resultList = ["不合适", "可以聊聊", "很合适"]
    percentTats = float(input("花在游戏上的时间比例？"))
    ffMiles = float(input("每年的飞行里数?"))
    iceCream = float(input("每年消耗的冰激凌数?"))
    inArr = np.array([ffMiles, percentTats, iceCream])
    datingDataMat, datingLabels = file2mattix("datingTestSet2.txt")
    normMat, ranges, minValue = autoNorm(datingDataMat)
    classifierResult = classify0((inArr - minValue) / ranges, normMat,
                                 datingLabels, 3)
    print("这个人对于你来说，是", resultList[classifierResult - 1], "的！")


# 将32X32的图片变为1X1024的向量
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

# 利用KNN算法对手写数字进行识别
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir("digits/trainingDigits")
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    # 将所有的训练集的图片都转换成向量，并将这些向量放到一个矩阵中
    for i in range(m):
        fileNameStr = trainingFileList[i]
        filename = fileNameStr.split(".")[0]
        classNumber = int(filename.split("_")[0])
        hwLabels.append(classNumber)
        trainingMat[i, :] = img2vector("digits/trainingDigits/" + fileNameStr)
    testFileList = listdir("digits/testDigits")
    mTest = len(testFileList)
    errorCount = 0.0
    # 利用测试集图片对KNN算法进行测试
    for i in range(mTest):
        testfileNameStr = testFileList[i]
        testfilename = testfileNameStr.split(".")[0]
        testVector = img2vector("digits/testDigits/" + testfileNameStr)
        classifierResult = classify0(testVector, trainingMat, hwLabels, 3)
        testclassNumber = int(testfilename.split("_")[0])
        print("KNN分类器预测的结果为：{}，真实的标签为：{}".format(classifierResult, testclassNumber))
        if testclassNumber != classifierResult:
            errorCount += 1.0
        print("KNN分类器在手写数字识别上的错误率为：{}%".format((errorCount/float(mTest)) * 100))


# 测试代码
# dataSet, labels = createDataSet()
# print(classify0([0, 0], dataSet, labels, 3))
# returnMat, classLabelVector = file2mattix("datingTestSet2.txt")
# print(classLabelVector[0: 20])
# print(returnMat[0:20, :])
# normDataSet, range, minValue = autoNorm(returnMat)
# print('range =', range)
# print('minValue =', minValue)
# print('normDataSet[0:20, :] = ', normDataSet[0:20, :])
# datingClassTest()
# classifyPerson()
# testVector = img2vector("digits/testDigits/0_0.txt")
# print(testVector[0, 0: 31])
# print(testVector[0, 32: 63])
handwritingClassTest()