# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 17:27:15 2017

@author: 第一组
"""

from pylab import mpl                #导入pylab用于设置默认字体，matplotlib默认不支持中文显示
from numpy import *
import time
import matplotlib.pyplot as plt
import numpy as np
import math
from tkinter.filedialog import *
from tkinter import *                #导入tkinter用于图ffgfgg形界面绘制
import struct  
UNCLASSIFIED = False
NOISE = 0
#*****************************K-means算法*****************************
# calculate Euclidean distance
def euclDistance(vector1, vector2):
	return sqrt(sum(power(vector2 - vector1, 2)))

# init centroids with random samples
def initCentroids(dataSet, k):
	numSamples, dim = dataSet.shape
	centroids = zeros((k, dim))
	for i in range(k):
		index = int(random.uniform(0, numSamples))
		centroids[i, :] = dataSet[index, :]
	return centroids

# k-means cluster
def kmeans(dataSet, k):
	numSamples = dataSet.shape[0]
	# first column stores which cluster this sample belongs to,
	# second column stores the error between this sample and its centroid
	clusterAssment = mat(zeros((numSamples, 2)))
	clusterChanged = True

	## step 1: init centroids
	centroids = initCentroids(dataSet, k)

	while clusterChanged:
		clusterChanged = False
		## for each sample
		for i in range(numSamples):
			minDist  = 100000.0
			minIndex = 0
			## for each centroid
			## step 2: find the centroid who is closest
			for j in range(k):
				distance = euclDistance(centroids[j, :], dataSet[i, :])
				if distance < minDist:
					minDist  = distance
					minIndex = j
			
			## step 3: update its cluster
			if clusterAssment[i, 0] != minIndex:
				clusterChanged = True
				clusterAssment[i, :] = minIndex, minDist**2

		## step 4: update centroids
		for j in range(k):
			pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]
			centroids[j, :] = mean(pointsInCluster, axis = 0)

	return centroids, clusterAssment
# show your cluster only available with 2-D data
def showCluster(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape


    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']


	# draw all samples
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
	# draw the centroids
    for i in range(k):
        plt.xlabel('x轴:月份')   
        plt.ylabel('y轴:机票价格')  
    plt.show()
def kmeansMain(filePath,K):
    ## step 1: load data
    dataSet = []
    fileIn = open(filePath)
    for line in fileIn.readlines():
        lineArr = line.strip().split('\t')
        dataSet.append([float(lineArr[0]), float(lineArr[1])])
## step 2: clustering...
    dataSet = mat(dataSet)
    k=K
    centroids, clusterAssment = kmeans(dataSet, k)

## step 3: show the result
    showCluster(dataSet, k, centroids, clusterAssment)
#*********************************DBSCAN算法******************************
def loadDataSet(fileName, splitChar='\t'):
    """
    输入：文件名
    输出：数据集
    描述：从文件读入数据集
    """
    dataSet = []
    with open(fileName) as fr:
        for line in fr.readlines():
            curline = line.strip().split(splitChar)
            fltline = list(map(float, curline))
            dataSet.append(fltline)
    return dataSet

def dist(a, b):
    """
    输入：向量A, 向量B
    输出：两个向量的欧式距离
    """
    return math.sqrt(np.power(a - b, 2).sum())

def eps_neighbor(a, b, eps):
    """
    输入：向量A, 向量B
    输出：是否在eps范围内
    """
    return dist(a, b) < eps

def region_query(data, pointId, eps):
    """
    输入：数据集, 查询点id, 半径大小
    输出：在eps范围内的点的id
    """
    nPoints = data.shape[1]
    seeds = []
    for i in range(nPoints):
        if eps_neighbor(data[:, pointId], data[:, i], eps):
            seeds.append(i)
    return seeds

def expand_cluster(data, clusterResult, pointId, clusterId, eps, minPts):
    """
    输入：数据集, 分类结果, 待分类点id, 簇id, 半径大小, 最小点个数
    输出：能否成功分类
    """
    seeds = region_query(data, pointId, eps)
    if len(seeds) < minPts: # 不满足minPts条件的为噪声点
        clusterResult[pointId] = NOISE
        return False
    else:
        clusterResult[pointId] = clusterId # 划分到该簇
        for seedId in seeds:
            clusterResult[seedId] = clusterId

        while len(seeds) > 0: # 持续扩张
            currentPoint = seeds[0]
            queryResults = region_query(data, currentPoint, eps)
            if len(queryResults) >= minPts:
                for i in range(len(queryResults)):
                    resultPoint = queryResults[i]
                    if clusterResult[resultPoint] == UNCLASSIFIED:
                        seeds.append(resultPoint)
                        clusterResult[resultPoint] = clusterId
                    elif clusterResult[resultPoint] == NOISE:
                        clusterResult[resultPoint] = clusterId
            seeds = seeds[1:]
        return True

def dbscan(data, eps, minPts):
    """
    输入：数据集, 半径大小, 最小点个数
    输出：分类簇id
    """
    clusterId = 1
    nPoints = data.shape[1]
    clusterResult = [UNCLASSIFIED] * nPoints
    for pointId in range(nPoints):
        point = data[:, pointId]
        if clusterResult[pointId] == UNCLASSIFIED:
            if expand_cluster(data, clusterResult, pointId, clusterId, eps, minPts):
                clusterId = clusterId + 1
    return clusterResult, clusterId - 1

def plotFeature(data, clusters, clusterNum):
    nPoints = data.shape[1]
    matClusters = np.mat(clusters).transpose()
    fig = plt.figure()
    scatterColors = ['red', 'green', 'blue', 'yellow', 'black', 'purple', 'orange', 'brown']
    ax = fig.add_subplot(111)
    for i in range(clusterNum + 1):
        colorSytle = scatterColors[i % len(scatterColors)]
        subCluster = data[:, np.nonzero(matClusters[:, 0].A == i)]
        ax.scatter(subCluster[0, :].flatten().A[0], subCluster[1, :].flatten().A[0], c=colorSytle, s=50)
    plt.xlabel('x轴:月份')   
    plt.ylabel('y轴:机票价格')  
    plt.show()

def dbscanMain(filePath,eps):
    dataSet = loadDataSet(filePath, splitChar='\t')
    dataSet = np.mat(dataSet).transpose()
    # print(dataSet)
    clusters, clusterNum = dbscan(dataSet, eps, 5)
    print("cluster Numbers = ", clusterNum)
    # print(clusters)
    plotFeature(dataSet, clusters, clusterNum)
#******************************************************UI界面****************************
def InitUI():
     global fileName
     global textBox                                      #声明消息栏UI
     global inputEpsBox                                     #声明输入栏UI
     global inputKBox
     initSampleInterval=Variable()                       #声明初始的输入栏中的采样间距值
     initSampleInterval.set("30")                         #将Eps设置为默认30
     
     initSampleK=Variable()                       #声明初始的输入栏中的K值
     initSampleK.set("3")                         #将K值默认设置为3
     root.title('MachineLearning Group1聚合函数绘制系统V1.0')
     
     textBox = Text(root, width='50', height='15')   
     
     textBox.insert('1.0', "                                  ***使用说明***\n")
     
     textBox.insert('2.0', "                           请通过按钮确定不同聚合函数的执行\n")
     
     textBox.insert('3.0', "               通过输入框点击回车确认K-Means的K值与Dbscan的的Eps值,否则取默认值\n")
     
     textBox.pack(fill=X)
     
     inputBoxLabel=Label(root, text="KBSCAN间距值设置")  
     
     inputBoxLabel.pack(side=LEFT)
     
     inputKBox=Entry(root,textvariable=initSampleK,background = 'green')
     inputKBox.pack(side=RIGHT)
     
     inputKLabel=Label(root, text="K-Means分类的K值设置") 
     inputKLabel.pack(side=RIGHT)
     
     
     inputEpsBox = Entry(root,textvariable=initSampleInterval,background = 'yellow')  #正确
     inputEpsBox.pack(side=LEFT)
     

     #InputBox.contents = StringVar()
     button = Button(root, text="Dbscan执行", command=DbscanDraw)
         
     button.pack(fill=X)
     
     buttonK=Button(root, text="K-MEANS执行", command=KmeansDraw)
     buttonK.pack(side=RIGHT)
def DbscanDraw():
    filePath = filedialog.askopenfilename(filetypes=[("所有文件", "*")])
    inputEps=int(inputEpsBox.get())
    dbscanMain(filePath,inputEps)
def KmeansDraw():
    filePath = filedialog.askopenfilename(filetypes=[("所有文件", "*")])
    inputK=int(inputKBox.get())
    kmeansMain(filePath,inputK)
if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = ['FangSong']      # 指定默认字体为仿宋
    mpl.rcParams['axes.unicode_minus'] = False          # 解决负号'-'显示为方块的问题
    root=Tk()
    InitUI()
    mainloop()
    
