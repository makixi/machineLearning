from math import log 

def calcShannonEnt(dataSet):
	numEntries=len(dataSet)
	labelCounts={}
	for featVec in dataSet:
		currentLabel=featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel]=0
		labelCounts[currentLabel]+=1
	shannonEnt=0.0
	for key in labelCounts:
		prob=float(labelCounts[key])/numEntries
		shannonEnt-=prob*log(prob,2)
	return shannonEnt

def createDataSet():
	dataSet=[[1,1,'yes'],
			[1,1,'yes'],
			[1,0,'no'],
			[0,1,'no'],
			[0,1,'no']]
	labels=['no surfacing','flippers']
	return dataSet,labels 

def splitDataSet(dataSet,axis,value):
	retDataSet=[]
	for featVec in dataSet:
		if featVec[axis]==value:
			reducedFeatVec=featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet

def chooseBestFeatureToSplit(dataSet):
	numFeatures=len(dataSet[0])-1
	baseEntropy=calcShannonEnt(dataSet)
	bestInfoGain=0.0
	bestFeature=-1
	for i in range(numFeatures):
		featList=[example[i] for example in dataSet]
		baseEntropy=calcShannonEnt(dataSet)
		uniqueVals=set(featList)
		newEntropy=0.0
		for value in uniqueVals:
			subDataSet=splitDataSet(dataSet,i,value)
			prob=len(subDataSet)/float(len(dataSet))
			newEntropy+=prob*calcShannonEnt(subDataSet)
		infoGain=baseEntropy-newEntropy
		if (infoGain>baseInfoGain):
			bestInfoGain=infoGain
			bestFeature=i
	return bestFeature

def classify(inputTree,featLabels,testVec):
	firstStr=inputTree.keys()[0]
	secondDict=inputTree[firstStr]
	featIndex=featLabels.index(firstStr)
	for key in secondDict.keys():
		if testVec[featIndex]==key:
			if type(secondDict[key]).__name__=='dict':
				classLabel=classify(secondDict[key],featLabels,testVec)
			else:
				classLabel=secondDict[key]
	return classLabel

myDat,labels=createDataSet()
print(calcShannonEnt(myDat))