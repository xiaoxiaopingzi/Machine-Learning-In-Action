# -*- coding: UTF-8 -*-
"""
@author: WanZhiWen 
@file: treeExplore.py 
@time: 2018-04-29 19:30

使用GUI来显示回归树
"""
from numpy import *
from tkinter import *
import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    import regTrees
except ImportError:
    raise ImportError('The file is not found. Please check the file name!')


def getInputs():
    try:
        # 通过tolNentry.get()获取tolN中值，注意需要判断用户输入的符号是否能转换为整数
        tolN = int(tolNentry.get())
    except:
        # 如果不能转换成整数，就弹出错误消息，并使用默认值10替换
        tolN = 10
        print("enter Integer for tolN")
        tolNentry.delete(0, END)
        tolNentry.insert(0, '10')
    try:
        tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print("enter Float for tolS")
        tolSentry.delete(0, END)
        tolSentry.insert(0, '1.0')
    return tolN, tolS


def reDraw(tolS, tolN):
    reDraw.f.clf()  # clear the figure
    reDraw.a = reDraw.f.add_subplot(111)
    if chkBtnVar.get():
        if tolN < 2: tolN = 2
        myTree = regTrees.createTree(reDraw.rawDat, regTrees.modelLeaf, \
                                     regTrees.modelErr, (tolS, tolN))
        yHat = regTrees.createForceCast(myTree, reDraw.testDat, \
                                       regTrees.modelTreeEval)
    else:
        myTree = regTrees.createTree(reDraw.rawDat, ops=(tolS, tolN))
        yHat = regTrees.createForceCast(myTree, reDraw.testDat)
    reDraw.a.scatter(reDraw.rawDat[:, 0].tolist(), reDraw.rawDat[:, 1].tolist(), s=5)  # use scatter for data set
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0)  # use plot for yHat
    reDraw.canvas.show()


def drawNewTree():
    tolN, tolS = getInputs()
    reDraw(tolN, tolS)


root = Tk()

# Label(root, text='Plot Place Holder').grid(row=0, columnspan=3)
reDraw.f = Figure(figsize=(5, 4), dpi=100)  # create canvas
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

Label(root, text="tolN").grid(row=1, column=0)
tolNentry = Entry(root)
tolNentry.grid(row=1, column=1)
tolNentry.insert(0, '10')
Label(root, text="tolS").grid(row=2, column=0)
tolSentry = Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0, '1.0')
Button(root, text="ReDraw", command=drawNewTree).grid(row=1, column=2, rowspan=3)
chkBtnVar = IntVar()
chkBtn = Checkbutton(root, text="Model Tree", variable=chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)

reDraw.rawDat = mat(regTrees.loadDataSet('sine.txt'))
reDraw.testDat = arange(min(reDraw.rawDat[:, 0]), max(reDraw.rawDat[:, 0]), 0.01)
reDraw(1.0, 10)

root.mainloop()
