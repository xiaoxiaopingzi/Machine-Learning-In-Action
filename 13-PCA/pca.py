# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: pca.py 
@time: 2018-05-03 10:59  
"""

import numpy as np
import matplotlib.pyplot as plt


# 装载数据集
def loadDataSet(fileName, delim='\t'):
    dataArr = []
    fr = [line.strip().split(delim) for line in open(fileName).readlines()]
    for i in range(len(fr)):
        temp = [float(i) for i in fr[i]]
        dataArr.append(temp)
    return np.mat(dataArr)


# 使用PCA对输入数据进行降维
def pca(dataset, topNfeat=9999999):
    # 对输入的数据进行均值清零
    meanValues = np.mean(dataset, axis=0)
    meanRemoved = dataset - meanValues
    # 获取数据的协方差矩阵，注意rowvar=False
    covmat = np.cov(meanRemoved, rowvar=False)
    # 获取协方差矩阵的特征向量和特征值
    eigVals, eigVectors = np.linalg.eig(np.mat(covmat))
    # 获取前k大的特征值对应的特征向量
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat + 1):-1]
    redEigVects = eigVectors[:, eigValInd]
    print('对应的特征向量为：')
    print(redEigVects)
    # 利用前k大的特征值对应的特征向量来进行PCA降维
    lowDDatMat = np.dot(meanRemoved, redEigVects)
    # 从降维之后的结果中恢复出原始的数据
    reconMat = np.dot(lowDDatMat, redEigVects.T) + meanValues
    return lowDDatMat, reconMat


# 将数据集中每列的NAN的值用每列的非NAN的值的均值替代
def replaceNanMean():
    dataMat = loadDataSet('secom.data', ' ')
    numFeater = dataMat.shape[1]
    for i in range(numFeater):
        meanVal = np.mean(dataMat[np.nonzero(~np.isnan(dataMat[:, i]))[0], i])
        dataMat[np.nonzero(np.isnan(dataMat[:, i]))[0], i] = meanVal
    return dataMat


# dataMat = loadDataSet('testSet.txt')
# lowMat, reconMat = pca(dataMat, 1)
# print(lowMat.shape)
# print(reconMat.shape)
# plt.scatter(dataMat[:, 0].tolist(), dataMat[:, 1].tolist(), marker='^', s=90)
# plt.scatter(reconMat[:, 0].tolist(), reconMat[:, 1].tolist(), marker='o', s=50, c='r')
# plt.show()
# plt.savefig('teseSet.png')
dataMat = replaceNanMean()
meanVals = np.mean(dataMat, axis=0)
meanRemoved = dataMat - meanVals
covMat = np.cov(meanRemoved, rowvar=False)
eigVals, eigVectors = np.linalg.eig(covMat)
print('特征值的个数为：', eigVals.shape[0])
print(eigVals)
