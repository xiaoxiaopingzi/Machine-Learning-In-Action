# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: kMeans.py 
@time: 2018-04-30 10:41  
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import urllib
import json
from time import sleep


# 从文件中装载数据
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = [float(i) for i in curLine]
        dataMat.append(fltLine)
    return dataMat


# 利用欧式距离计算两个向量之间的距离
def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.square(vecA - vecB)))


# 根据给定数据集构建一个包含k个随机质心和集合
def randCent(dataSet, k):
    n = dataSet.shape[1]
    centroids = np.mat(np.zeros((k, n)))
    for i in range(n):
        # 随机质心必须位于整个数据集的边界之内
        minValue = float(min(dataSet[:, i]))
        rangeJ = float(max(dataSet[:, i]) - minValue)
        centroids[:, i] = minValue + rangeJ * np.random.rand(k, 1)
    return centroids


# 根据给定数据集构建一个包含k个随机质心和集合，此方法从数据集中随机选取k个样本点作为随机的质心
def randCent2(dataSet, k):
    m, n = dataSet.shape
    randomIndex = np.random.choice(range(m), k)
    return dataSet[randomIndex]


# k-means聚类
def kMeans(dataSet, k, distmeans=distEclud, createCent=randCent):
    m = dataSet.shape[0]
    clusterAssment = np.zeros((m, 2))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = float('inf')
            minIndex = -1
            for j in range(k):
                distJ = distmeans(dataSet[i, :], centroids[j, :])
                if distJ < minDist:
                    minIndex = j
                    minDist = distJ
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, np.square(minDist)
        # print('当前的质心为：', centroids)
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0] == cent)]
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment


# 二分k-means聚类
def biKmeans(dataSet, k, distmeas=distEclud):
    m = dataSet.shape[0]
    clusterAssiment = np.zeros((m, 2))
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssiment[j, 1] = np.square(distmeas(np.mat(centroid0), dataSet[j, :]))
    while (len(centList) < k):
        lowestSSE = float('inf')
        bestCentToSplit = -1
        bestNewCents = None
        bsetClustAss = None
        for i in range(len(centList)):
            ptsINcurrCluster = dataSet[np.nonzero(clusterAssiment[:, 0] == i)[0], :]
            print(clusterAssiment)
            centroidMat, splitClusterAss = kMeans(ptsINcurrCluster, 2, distmeas)
            sseSplit = np.sum(splitClusterAss[:, 1])
            sseNotSplit = np.sum(clusterAssiment[np.nonzero(clusterAssiment[:, 0] != i)[0], 1])
            print('sseSplit and sseNotSplit:', sseSplit, sseNotSplit)
            if (sseNotSplit + sseSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bsetClustAss = splitClusterAss.copy()
                lowestSSE = sseNotSplit + sseSplit
        print('The bestCentToSplit is: ', bestCentToSplit)
        print('The len of bestClustAss is:', len(bsetClustAss))
        bsetClustAss[np.nonzero(bsetClustAss[:, 0] == 0)[0], 0] = bestCentToSplit
        bsetClustAss[np.nonzero(bsetClustAss[:, 0] == 1)[0], 0] = len(centList)
        centList[bestCentToSplit] = bestNewCents[0, :]
        centList.append(bestNewCents[1, :])
        clusterAssiment[np.nonzero(clusterAssiment[:, 0] == bestCentToSplit)[0], :] = bsetClustAss
    return centList[0], clusterAssiment


def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  # create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'  # JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.urlencode(params)
    yahooApi = apiStem + url_params  # print url_params
    print('yahooApi = ', yahooApi)
    c = urllib.urlopen(yahooApi)
    return json.loads(c.read())


def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print("%s\t%f\t%f" % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else:
            print("error fetching")
        sleep(1)
    fw.close()


def distSLC(vecA, vecB):
    a = np.sin(vecA[0, 1] * np.pi / 180) * np.sin(vecB[0, 1] * np.pi / 180)
    b = np.cos(vecA[0, 1] * np.pi / 180) * np.cos(vecB[0, 1] * np.pi / 180) \
        * np.cos(np.pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return np.arccos(a + b) * 6371.0


def clusterClubs(numClust=5):
    dataList = []
    for line in open('places.txt').readlines():
        lineArr = line.strip().split('\t')
        dataList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = np.mat(dataList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distmeas=distSLC)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[np.nonzero(clustAssing[:, 0] == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].tolist(), ptsInCurrCluster[:, 1].tolist(), marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:, 0].tolist(), myCentroids[:, 1].tolist(), marker='+', s=300)
    plt.show()


# dataMat = np.mat(loadDataSet('testSet.txt'))
# print(distEclud(dataMat[0], dataMat[1]))
# print(min(dataMat[:, 0]))
# print(min(dataMat[:, 1]))
# print(max(dataMat[:, 0]))
# print(max(dataMat[:, 1]))
# print(randCent(dataMat, 2))
# kMeans(dataMat, 4)

# dataMat3 = np.mat(loadDataSet('testSet2.txt'))
# MycentList, myNewAssments = biKmeans(dataMat3, 3)
# print(MycentList)
clusterClubs(5)
# dataMat = np.mat(loadDataSet('testSet.txt'))
# print(randCent(dataMat, 5))