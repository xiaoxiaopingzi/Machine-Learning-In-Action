# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: bayes.py 
@time: 2018-04-22 13:55

从文本中构造词向量
"""
import numpy as np
import random
import re
import feedparser
import operator


# 生成数据集
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1表示侮辱性言论，0表示正常言论
    return postingList, classVec


# 根据数据集创建词汇表
def createVocabList(dataSet):
    vocabSet = set()  # 使用set集合来去重
    for line in dataSet:
        vocabSet = vocabSet | set(line)  # |表示集合的并集
    # 将set集合转换为list，以便使用list集合的index方法
    return sorted(list(vocabSet))


# 根据词汇表将输入的句子转换为向量,伯努利模型
def setOfWord2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 创造一个和词汇表等长的全0向量
    for word in inputSet:
        if word in vocabList:
            # vocabList.index(word)用于找到在vocabList集合中word第一次出现的位置的索引
            returnVec[vocabList.index(word)] = 1
        else:
            print("The word: {} is not in my vocabulary!".format(word))
    return returnVec


# 根据词汇表将输入的句子转换为向量,多项式模型
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 创造一个和词汇表等长的全0向量
    for word in inputSet:
        if word in vocabList:
            # vocabList.index(word)用于找到在vocabList集合中word第一次出现的位置的索引
            returnVec[vocabList.index(word)] += 1
    return returnVec


# 朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix, trainCategory):
    m = len(trainCategory)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / m  # 得到侮辱性文档的比例
    p0Num = np.ones(numWords)  # 防止出现概率为0的情况，所以初始值设为1
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(m):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        if trainCategory[i] == 0:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 为了防止下溢的情况，使用log()函数来防止下溢
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

# 根据训练好的朴素贝叶斯模型来进行分类
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

# 测试朴素贝叶斯模型
def testingNB():
    listOfPosts, listCLasses = loadDataSet()
    labelMeaning = ['非侮辱性言论', '侮辱性言论']
    myVocabList = createVocabList(listOfPosts)
    print('词汇表为：', myVocabList)
    trainMat = []
    for postinDoc in listOfPosts:
        trainMat.append(setOfWord2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, listCLasses)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWord2Vec(myVocabList, testEntry))
    result = classifyNB(thisDoc, p0V, p1V, pAb)
    print(testEntry, "classify as: {}({})".format(result, labelMeaning[result]))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWord2Vec(myVocabList, testEntry))
    result = classifyNB(thisDoc, p0V, p1V, pAb)
    print(testEntry, "classify as: {}({})".format(result, labelMeaning[result]))

# 利用正则表达式解析邮件
def textparse(bigString):
    # \W表示数字和字母
    regEx = re.compile('\\W*')
    listOfTokens = regEx.split(bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

# 利用朴素贝叶斯模型来对垃圾邮件进行分类，并测试模型的错误率
def spamTest():
    docList = []
    classList = []
    fullTest = []
    for i in range(1, 26):
        wordList = textparse(open("email/spam/{}.txt".format(i)).read())
        docList.append(wordList)
        fullTest.extend(wordList)
        classList.append(1)
        wordList = textparse(open("email/ham/{}.txt".format(i)).read())
        docList.append(wordList)
        fullTest.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    # 随机构造训练集和测试集
    trainingSet = list(range(50))  # trainingSet是一个整数列表
    testSet = []
    for i in range(10):
        # 随机选取测试集的索引
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []
    trainClass = []
    for docIndex in trainingSet:
        trainMat.append(setOfWord2Vec(vocabList, docList[docIndex]))
        trainClass.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClass))
    errorCount = 0.0
    for i in testSet:
        wordVector = setOfWord2Vec(vocabList, docList[i])
        result = classifyNB(wordVector, p0V, p1V, pSpam)
        if result != classList[i]:
            errorCount += 1
            if result == 0:
                print('邮件：', docList[i], '预测为：正常邮件，但实际是垃圾邮件!')
            else:
                print('邮件：', docList[i], '预测为：垃圾邮件，但实际是正常邮件!')
    print('The error rate is:', float(errorCount) / len(testSet))


# 计算在fullText中每个单词出现的次数，并按照这个次数按从大到小进行排序
def calcMostFreq(vocabList, fullText):
    freqDict = {}
    for word in vocabList:
        freqDict[word] = fullText.count(word)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


def localWords(feed1, feed0):
    docList = []
    classList = []
    fullTest = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textparse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullTest.extend(wordList)
        classList.append(1)
        wordList = textparse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullTest.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    # 去掉出现次数最高的30个单词，还可以去掉停止词
    top30Words = calcMostFreq(vocabList, fullTest)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    # 随机构造训练集和测试集
    trainingSet = list(range(2 * minLen))  # trainingSet是一个整数列表
    testSet = []
    for i in range(20):
        # 随机选取测试集的索引
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []
    trainClass = []
    for docIndex in trainingSet:
        trainMat.append(setOfWord2Vec(vocabList, docList[docIndex]))
        trainClass.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClass))
    errorCount = 0.0
    for i in testSet:
        wordVector = setOfWord2Vec(vocabList, docList[i])
        result = classifyNB(wordVector, p0V, p1V, pSpam)
        if result != classList[i]:
            errorCount += 1
            if result == 0:
                print('RSS：', docList[i], '预测为：feed0，但实际是feed1!')
            else:
                print('RSS：', docList[i], '预测为：feed1，但实际是feed0!')
    print('The error rate is:', float(errorCount) / len(testSet))
    return vocabList, p0V, p1V


def getTopWords(ny,sf):
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])

# spamTest()
# 这里的RSS信息源无法使用
ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
# print(ny['entries'])
# print(len(ny['entries']))
# localWords(ny, sf)
getTopWords(ny,sf)