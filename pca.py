from numpy import *

def loadDataSet(fileName,delim='\t'):
	fr=open(fileName)
	stringArr=[line.strip().split(delim) for line in fr.readlines()]
	datArr=[map(float,line) for line in stringArr]
	return mat(datArr)

def pca(dataMat,topNfeat=9999999):
	meanVals=mean(dataMat,axis=0)
	meanRemoved=dataMat-meanVals	#去平均值
	covMat=cov(meanRemoved,rowvar=0)
	eigVals,eigVects=linalg.eig(mat(covMat))
	eigValInd=argsort(eigVals)
	eigValInd=eigValInd[:-(topNfeat+1):-1]
	redEigVects=eigVects[:,eigValInd]	#排序
	lowDataMat=meanRemoved*redEigVects	#将数据转换到新空间
	reconMat=(lowDataMat*redEigVects.T)+meanVals
	return lowDataMat,reconMat