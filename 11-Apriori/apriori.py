# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: apriori.py 
@time: 2018-05-01 10:30  
"""


def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


# 根据数据集创建只包含一项的项集
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if [item] not in C1:
                C1.append([item])
    C1.sort()
    return [frozenset(i) for i in C1]


# 对项集CK进行频繁项集过滤，小于最小支持度的项集会被过滤掉
def scanD(D, CK, minSupport):
    ssCnt = {}
    for tid in D:
        for can in CK:
            if can.issubset(tid):
                if can not in ssCnt.keys():
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = len(D)
    retList = []
    # supportData用于保存每个项集的支持度
    supportData = {}
    for key in ssCnt.keys():
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData


# 根据频繁项集LK生成新的k项的项集
def aprioriGen(LK, k):  # create CK
    retList = []
    lenLK = len(LK)
    for i in range(lenLK):
        for j in range(i + 1, lenLK):
            L1 = list(LK[i])[: k - 2]
            L2 = list(LK[j])[: k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(LK[i] | LK[j])
    return retList


# apriori算法，整合前面的函数得到整个的apriori算法
def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)
    L1, supportData = scanD(dataSet, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k - 2]) > 0):
        CK = aprioriGen(L[k - 2], k)
        LK, supK = scanD(dataSet, CK, minSupport)
        supportData.update(supK)
        L.append(LK)
        k += 1
    return L, supportData


# 根据频繁项集生成关联规则
def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


def calcConf(freqSet, H, supportData, br1, minConf=0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            print(freqSet - conseq, '--->', conseq, 'conf:', conf)
            br1.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


def rulesFromConseq(freqSet, H, supportData, br1, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)):
        Hmp1 = aprioriGen(H, m + 1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, br1, minConf)
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, br1, minConf)


dataSet = loadDataSet()
print(dataSet)
# C1 = createC1(dataSet)
# print('C1 = ', C1)
# L1, support = scanD(dataSet, C1, 0.5)
# print('L1 = ', L1)
# print(support)
L, suppData = apriori(dataSet, minSupport=0.5)
print('所有的频繁项集为：', L)
# print(suppData)
# print(aprioriGen(L[0], 2))
rules = generateRules(L, suppData, minConf=0.5)
