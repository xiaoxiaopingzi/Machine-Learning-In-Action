# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: mrMeanReducer.py 
@time: 2018-05-05 13:21  
"""

import sys

def read_input(file):
    for line in file:
        # rstrip()表示返回字符串line的副本，并删除尾随空白
        yield line.rstrip()


input = read_input(sys.stdin)
mapperOut = [line.split(' ') for line in input]
cumVal = 0.0
cumSumSq = 0.0
cumN = 0.0
for instance in mapperOut:
    nj = float(instance[0])
    cumN += nj
    cumVal += nj * float(instance[1])
    cumSumSq += nj * float(instance[2])
mean = cumVal / cumN
varSum = (cumSumSq - 2 * mean * cumVal + cumN * mean * mean) / cumN
print(cumN, mean, varSum)
print(sys.stdout, 'report: still alive')