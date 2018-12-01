from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense,Embedding
from keras.layers import LSTM
from keras.datasets import imdb

#最多使用的单词数
max_features=20000
#循环神经网络的截断长度
maxlen=80
batch_size=32
(trainX,trainY),(testX,testY)=imdb.load_data(num_words=max_features)
print(len(trainX),'train sequences')
print(len(testX),'test sequences')

#cnn的循环长度是固定的，这里需要先将所有段落都固定成统一长度
#对于长度不够的段落，使用默认值0来填充
#超过长度的段落直接忽略掉超过部分
trainX=sequence.pad_sequences(trainX,maxlen=maxlen)

#完成数据预处理之后构建模型
model=Sequential()
model.add(Embedding(max_features,128))
model.add(LSTM(128,dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(trainX,trainY,batch_size=batch_size,epochs=15,validation_data=(testX,testY))
score=model.evaluate(testX,testY,batch_size=batch_size)
print(score[0])#test loss
print(score[1])#test accuracy