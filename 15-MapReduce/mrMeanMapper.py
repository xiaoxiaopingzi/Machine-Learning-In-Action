# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: mrMeanMapper.py 
@time: 2018-05-05 10:19  
"""

import sys
import numpy as np


def read_input(file):
    for line in file:
        yield line.rstrip()


input = read_input(sys.stdin)
input = [float(line) for line in input]
numInputs = len(input)
input = np.mat(input)
sqInput = np.square(input)
print(numInputs, np.mean(input), np.mean(sqInput))
# print(sys.stdout, 'report: still alive')
