# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: svmMLiA.py 
@time: 2018-04-25 14:53  
"""
import numpy as np


# 从文本文件中装载数据集
def loadDataset(fileName):
    mat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        mat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return mat, labelMat


def selectJrand(i, m):
    j = i
    # 这个while循环保证返回的j和i是不相等的
    while j == i:
        j = np.random.randint(0, m)
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).T
    m, n = dataMatrix.shape
    alphas = np.zeros((m, 1))
    b = 0
    iter = 0
    while iter < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            fxi = np.dot(np.multiply(alphas, labelMat).T,
                         np.dot(dataMatrix, dataMatrix[i, :].T)) + b
            Ei = float(fxi) - float(labelMat[i])
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or \
                    ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fxj = np.dot(np.multiply(alphas, labelMat).T,
                             np.dot(dataMatrix, dataMatrix[j, :].T)) + b
                Ej = float(fxj) - float(labelMat[j])
                alphaIold = alphas[i, 0]
                alphaJold = alphas[j, 0]
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print('L == H')
                    continue
                # eta是alpha[j]的最优修改量
                eta = 2.0 * np.dot(dataMatrix[i, :], dataMatrix[j, :].T) - \
                      np.dot(dataMatrix[i, :], dataMatrix[i, :].T) - \
                      np.dot(dataMatrix[j, :], dataMatrix[j, :].T)
                if eta >= 0:
                    print("eta>=0")
                    continue
                alphas[j, 0] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue
                # update i by the same amount as j
                alphas[i, 0] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                # the update is in the oppostie direction
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * \
                     np.dot(dataMatrix[i, :], dataMatrix[i, :].T) - labelMat[j] * \
                     (alphas[j] - alphaJold) * np.dot(dataMatrix[i, :], dataMatrix[j, :].T)
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * \
                     np.dot(dataMatrix[i, :], dataMatrix[j, :].T) - labelMat[j] * \
                     (alphas[j] - alphaJold) * np.dot(dataMatrix[j, :], dataMatrix[j, :].T)
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas


if __name__ == '__main__':
    dataArr, labelArr = loadDataset('testSet.txt')
    print(labelArr)
    b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    print('b = ', b)
    print(alphas[alphas > 0])
    for i in range(100):
        if alphas[i] > 0:
            print(dataArr[i], "  ", labelArr[i])
