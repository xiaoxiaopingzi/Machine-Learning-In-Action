# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: svdRec.py 
@time: 2018-05-03 15:18  
"""
import numpy as np
from numpy import linalg as la


# 装载数据集
def loadExData():
    return [[0, 0, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 1, 0, 0]]


def loadExData2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


# 相似度计算方式一，使用向量之间的欧式距离
def euclidSim(inA, inB):
    return 1 / (1 + la.norm(inA - inB))


# 相似度计算方式二，使用皮尔逊系数
def pearsSim(inA, inB):
    if len(inB) < 3:
        return 1.0
    else:
        return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=False)[0][1]


# 相似度计算方式三，使用余弦相似度
def cosSim(inA, inB):
    num = float(np.dot(inA.T, inB))
    denom = la.norm(inB) * la.norm(inA)
    return 0.5 + 0.5 * (num / denom)


# 预测用户user对第item道菜的评分——根据用户所有已经进行了评分的菜来预测用户对item的评分
def standEst(dataMat, user, simMeas, item):
    n = dataMat.shape[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:  # 如果第j道菜，用户未进行评分，则直接跳过
            continue
        else:
            # overlap表示对第j个物品以及第item个物品都进行了评分的用户的索引
            overLap = np.nonzero(np.logical_and(dataMat[:, j] > 0, dataMat[:, item] > 0))[0]
            if len(overLap) == 0:
                similarity = 0
            else:
                similarity = simMeas(dataMat[overLap, j], dataMat[overLap, item])
            # print('物品{}和{}的相似度为：{}'.format(j, item, similarity))
            simTotal += similarity
            ratSimTotal += userRating * similarity
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


# 利用SVD来预测用户评分
def svdEst(dataMat, user, simMeas, item):
    n = dataMat.shape[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U, Sigma, VT = la.svd(dataMat)
    Sig4 = np.mat(np.eye(4) * Sigma[:4])  # arrange Sig4 into a diagonal matrix
    # 利用U矩阵将物品转换到低维空间中
    xformedItems = dataMat.T * U[:, :4] * Sig4.I  # create transformed items
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue
        else:
            # 直接在低维空间中计算第j道菜和第item道菜的相似度
            similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
            print('the %d and %d similarity is: %f' % (item, j, similarity))
            simTotal += similarity
            ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


# 对用户user进行菜品的推荐，返回用户所有未点评的菜的预测的评分，按照预测的评分高低返回
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = np.nonzero(dataMat[user, :] == 0)[1]
    if len(unratedItems) == 0:
        return 'you rated everything'
    else:
        itemScores = []
        for item in unratedItems:
            estimatedScore = estMethod(dataMat, user, simMeas, item)
            itemScores.append((item, estimatedScore))
        returnValue = sorted(itemScores, key=lambda p: p[1], reverse=True)[:N]
        return returnValue


# 打印出一个32 X 32的矩阵
def printMat(inMat, thresh=0.8):
    for i in range(32):
        for j in range(32):
            if float(inMat[i, j]) > thresh:
                print(1, end=" ")  # 这样打印不会换行，只会输出一个空格
            else:
                print(0, end=" ")
        print()


# 利用SVD对图像进行压缩
def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = np.mat(myl)
    print('--------------原始的矩阵为:-----------------')
    printMat(myMat, thresh)
    U, sigma, VT = la.svd(myMat)
    sigRecon = np.zeros((numSV, numSV))
    for i in range(numSV):
        sigRecon[i, i] = sigma[i]
    # 只取前numSV个奇异值来对原始的矩阵进行重构
    reconMat = np.dot(np.dot(U[:, :numSV], sigRecon), VT[:numSV, :])
    print('--------------压缩后的矩阵为:-----------------')
    printMat(reconMat, thresh)


myMat = np.mat(loadExData())
# print(euclidSim(myMat[:, 0], myMat[:, 4]))
# print(euclidSim(myMat[:, 0], myMat[:, 0]))
# print('--------------------------------')
# print(pearsSim(myMat[:, 0], myMat[:, 4]))
# print(pearsSim(myMat[:, 0], myMat[:, 0]))
# print('--------------------------------')
# print(cosSim(myMat[:, 0], myMat[:, 4]))
# print(cosSim(myMat[:, 0], myMat[:, 0]))
myMat[0, 1] = myMat[0, 0] = myMat[1, 0] = myMat[2, 0] = 4
myMat[3, 3] = 2
print(myMat)
print(recommend(myMat, 2))
print(recommend(myMat, 2, estMethod=svdEst))
print(recommend(myMat, 1, estMethod=svdEst))
# temp = np.mat(np.eye(4) * [1, 2, 3, 4])
# print(temp.I)
imgCompress(numSV=2)
