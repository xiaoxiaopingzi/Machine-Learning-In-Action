# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: fpGrowth.py 
@time: 2018-05-02 10:17  
"""


# FP树的节点对象
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur

    # 打印出FP树，根据ind的大小来确定每行开头的空格数
    def disp(self, ind=1):
        print(' ' * ind, self.name, '   ', self.count)
        for child in self.children.values():
            # 使用递归来打印出每个子节点
            child.disp(ind + 1)


# 装载数据集
def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat


# 将list形式的数据集转换为dict形式的数据集
def createInitSet(dataSet):
    reDict = {}
    for trans in dataSet:
        reDict[frozenset(trans)] = 1
    return reDict


# 创造一颗FP树，dataSet为数据集，minSup为最小支持度
def createTree(dataSet, minSup=1):
    headerTable = {}  # headerTable的key表示所有的单个元素的项集，value表示对应的key出现的次数
    # 获取所有的单个元素的支持度
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    # 移除小于最小支持度的元素项
    # dict对象不允许在循环时删除元素，需要将其转换为list
    for key in list(headerTable.keys()):
        if headerTable[key] < minSup:
            del (headerTable[key])
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:
        return None, None
    for key in headerTable.keys():
        headerTable[key] = [headerTable[key], None]
    retTree = treeNode('Null Set', 1, None)
    for tranSet, count in dataSet.items():
        localD = {}
        # 去掉tranSet中的非频繁项集
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            # sorted(localD.items(), key=lambda p: p[1], reverse=True)表示对字典按照value进行排序
            # localD.items()表示取字典dict的键值对，可以通过v[0],v[1]分别取每个键值对的key和value
            orderItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            updateTree(orderItems, retTree, headerTable, count)
    return retTree, headerTable


def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] is None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        # items[1::]表示取索引为1以及1以后的所有位置的元素,注意一定要写成items[1::],
        # 不能写成items[1:]，否则会出现错误
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


def updateHeader(nodeToTest, targetNode):
    # 通过循环来找到链表的末尾的元素
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    # 将链表的末尾设定为新的targetNode
    nodeToTest.nodeLink = targetNode


# 根据leafNode对FP树进行上溯，将上溯的结果保存在prefixPath列表中
def ascendTree(leafNode, prefixPath):
    if leafNode.parent is not None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


# 发现以basePat结尾的所有路径
def findPrefixPath(basePat, treeNode):
    condPAts = {}
    while treeNode is not None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPAts[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPAts


# 递归查找频繁项集
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        myCondTree, myHead = createTree(condPattBases, minSup)
        if myHead is not None:
            print('conditional tree for:', newFreqSet)
            myCondTree.disp()
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)


# import twitter
from time import sleep
import re


def textParse(bigString):
    # 使用正则表达式去除URL
    urlsRemoved = re.sub('(http:[/][/]|www.)([a-z]|[A-Z]|[0-9]|[/.]|[~])*', '', bigString)
    # 获取推文中的单词，去掉符号、空格等
    listOfTokens = re.split(r'\W*', urlsRemoved)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def getLotsOfTweets(searchStr):
    CONSUMER_KEY = ''
    CONSUMER_SECRET = ''
    ACCESS_TOKEN_KEY = ''
    ACCESS_TOKEN_SECRET = ''
    api = twitter.Api(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET,
                      access_token_key=ACCESS_TOKEN_KEY,
                      access_token_secret=ACCESS_TOKEN_SECRET)
    # you can get 1500 results 15 pages * 100 per page
    resultsPages = []
    for i in range(1, 15):
        print("fetching page %d" % i)
        searchResults = api.GetSearch(searchStr, per_page=100, page=i)
        resultsPages.append(searchResults)
        sleep(6)
    return resultsPages


def mineTweets(tweetArr, minSup=5):
    parsedList = []
    for i in range(14):
        for j in range(100):
            parsedList.append(textParse(tweetArr[i][j].text))
    initSet = createInitSet(parsedList)
    myFPtree, myHeaderTab = createTree(initSet, minSup)
    myFreqList = []
    mineTree(myFPtree, myHeaderTab, minSup, set(), myFreqList)
    return myFreqList


# rootNode = treeNode('pyramid', 9, None)
# rootNode.children['eye'] = treeNode('eye', 13, None)
# rootNode.children['phoenix'] = treeNode('phoenix', 3, None)
# rootNode.disp()
# localD = {'a': 1, 'b': 2, 'c': 3}
# for v in sorted(localD.items(), key=lambda p: p[1], reverse=True):
#     print(v[1])
# simpDat = loadSimpDat()
# initSet = createInitSet(simpDat)
# print(initSet)
# myFPtree, myHeaderTab = createTree(initSet, 3)
# myFPtree.disp()
# print(findPrefixPath('r', myHeaderTab['r'][1]))
# print(findPrefixPath('z', myHeaderTab['z'][1]))
# print(findPrefixPath('x', myHeaderTab['x'][1]))
# print(myHeaderTab)
# freqItems用于保存生成的频繁项集
# freqItems = []
# mineTree(myFPtree, myHeaderTab, 3, set(), freqItems)
# print(freqItems)


parseDat = [line.split() for line in open('kosarak.dat').readlines()]
initSet = createInitSet(parseDat)
myFPTree, myHeadTab = createTree(initSet, 100000)
myFreqList = []
mineTree(myFPTree, myHeadTab, 100000, set(), myFreqList)
print(myFreqList)
print(len(myFreqList))