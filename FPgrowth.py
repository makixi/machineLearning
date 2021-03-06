class treeNode:
	def __init__(self,nameValue,numOccur,parentNode):
		self.name=nameValue
		self.count=numOccur
		self.nodeLink=None 
		self.parent=parentNode
		self.children={}

	def inc(self,numOccur):
		self.count+=numOccur

	def disp(self,ind=1):
		print(' '*ind,self.name,' ',self.count)


def createTree(dataSet,minSup=1):
	headerTable={}

	# 第一次遍历扫描数据集并统计每个元素项出现的频度。这些信息被保存在头指针中。
	for trans in dataSet:
		for item in trans:
			headerTable[item]=headerTable.get(item,0)+dataSet[trans]

	for k in list(headerTable.keys()):
		if headerTable[k]<minSup:
			del(headerTable[k])	#移除不满足最小支持度的元素项

	freqItemSet=set(headerTable.keys())

	# 如果所有项都不频繁，无需下一步处理
	if len(freqItemSet)==0:
		return None,None

	# 对头指针表稍加扩展以便可以保存计数值及指向每种类型第一个元素项的指针
	for k in headerTable:
		headerTable[k]=[headerTable[k],None]

	# 根节点
	retTree=treeNode('Null Set',1,None)

	for tranSet,count in dataSet.items():
		localD={}
		for item in tranSet:
			for item in freqItemSet:
				localD[item]=headerTable[item][0]	#根据全局频率对每个事务中的元素进行排序
		if len(localD)>0:
			orderedItems=[v[0] for v in sorted(localD.items(),key=lambda p: p[1],reverse=True)]
			updateTree(orderedItems,retTree,headerTable,count)	#使用排序后的频率项集对树进行填充
		return retTree,headerTable

def updateTree(items,inTree,headerTable,count):
	# 首先测试事务中的第一个元素项是否作为子节点存在。
	if items[0] in inTree.children:
		# 如果存在，则更新该元素项的计数
		inTree.children[items[0]].inc(count)
	else:
		# 如果不存在，则创建一个新的treeNode并将其作为一个子节点添加到树中，这时，头指针表也要更新以指向新的节点。
		inTree.children[items[0]]=treeNode(items[0],count,inTree)
		if headerTable[items[0]][1]==None:
			headerTable[items[0]][1]=inTree.children[items[0]]
		else:
			# 更新头指针表需要调用函数updateHeader
			updateHeader(headerTable[items[0]][1],inTree.children[items[0]])
	if len(items)>1:
		updateTree(items[1::],inTree.children[items[0]],headerTable,count)


# 确保节点链接指向树中该元素项的每一个实例，从头指针的nodeLink开始，一直沿着nodeLink直到到达链表末尾。
# 当处理树的时候，一种自然的反应就是迭代完整每一件事。当以相同方式处理链表时可能会遇到一些问题，
# 原因是如果链表很长可能会遇到迭代调用的次数限制
def updateHeader(nodeToTest,targetNode):
	while(nodeToTest.nodeLink!=None):
		nodeToTest=nodeToTest.nodeLink
	nodeToTest.nodeLink=targetNode

def loadSimpDat():
	simpDat=[['r','z','h','j','p'],
			['z','y','x','w','v','u','t','s'],
			['z'],
			['r','x','n','o','s'],
			['y','r','x','z','q','t','p'],
			['y','z','x','e','q','s','t','m']]
	return simpDat

def createInitSet(dataSet):
	retDict={}
	for trans in dataSet:
		retDict[frozenset(trans)]=1
	return retDict

def ascendTree(leafNode,prefixPath):
	#迭代上溯整棵树
	if leafNode.parent!=None:
		prefixPath.append(leafNode)
		ascendTree(leafNode.parent,prefixPath)

# 遍历链表直到到达结尾。每遇到一个元素项都会调用ascendTree()来上溯FP树，并收集所有遇到的元素项的名称。
# 该列表返回之后添加到条件模式基字典condPats中
def findPrefixPath(basePat,treeNode):
	condPats={}
	while treeNode!=None:
		prefixPath=[]
		ascendTree(treeNode,prefixPath)
		if len(prefixPath)>1:
			condPats[frozenset(prefixPath[1:])]=treeNode.count 
		treeNode=treeNode.nodeLink
	return condPats

#递归查找频繁项集
def minTree(inTree,headerTable,minSup,preFix,freqItemList):
	# 对头指针表中元素项按照其出现频率进行排序，默认是从小到大
	bigL=[v[0] for v in sorted(headerTable.items(),key=lambda p:p[1])]
	# 默认是从小到大，下面过程是从头指针的底端开始
	for basePat in bigL:
		newFreqSet=preFix.copy()
		newFreqSet.add(basePat)
		# 将每个频繁项添加到频繁项集列表freqItemList中
		freqItemList.append(newFreqSet)
		# 使用findPrefixPath()创建条件基
		condPatBases=findPrefixPath(basePat,headerTable[basePat][1])
		# 将条件基condPattBases作为新数据集传递给createTree()函数
        # 这里为函数createTree()添加足够的灵活性，确保它可以被重用于构建条件树
		myCondTree,myHead=createTree(condPatBases,minSup)
		# 如果树中有元素项的话，递归调用mineTree()函数
		if myHead!=None:
			minTree(myCondTree,myHead,minSup,newFreqSet,freqItemList)


# rootNode = treeNode('pyramid',9,None)
# rootNode.children['eye'] = treeNode('eye',13,None)
# a = rootNode.disp()
# print(a)

# simpDat=loadSimpDat()
# initSet=createInitSet(simpDat)
# myFPtree,myHeaderTab=createTree(initSet,3)
# print(myFPtree.disp())

