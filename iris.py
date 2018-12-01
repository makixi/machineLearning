from sklearn import model_selection
from sklearn import datasets
from sklearn import metrics
import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat

# 导入TFlearn
learn = tf.contrib.learn


# 自定义softmax回归模型：在给定输入的训练集、训练集标签，返回在这些输入上的预测值、损失值以及训练步骤
def my_model(features, target):
    # 将预测目标转化为one-hot编码的形式，共有三个类别，所以向量长度为3
    target = tf.one_hot(target, 3, 1, 0)
    # 计算预测值及损失函数：封装了一个单层全连接的神经网络
    logits = tf.contrib.layers.fully_connected(features, 3, tf.nn.softmax)
    loss = tf.losses.softmax_cross_entropy(target, logits)

    # 创建模型的优化器，并优化步骤
    train_op = tf.contrib.layers.optimize_loss(
        loss,  # 损失函数
        tf.contrib.framework.get_global_step(),  # 获取训练步骤并在训练时更新
        optimizer='Adam',  # 定义优化器
        learning_rate=0.01)  # 定义学习率

    # 返回给定数据集上的预测结果、损失值以及优化步骤
    return tf.arg_max(logits, 1), loss, train_op


# 加载iris数据集，并划分为训练集和测试集
iris = datasets.load_iris()
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=0)

# 将数据转化成TensorFlow要求的float32格式
x_train, x_test = map(np.float32, [x_train, x_test])

# 封装和训练模型，输出准确率
classifier = SKCompat(learn.Estimator(model_fn=my_model, model_dir="Models/model_1"))
classifier.fit(x_train, y_train, steps=200)

# 预测
y_predicted = [i for i in classifier.predict(x_test)]
# 计算准确度
score = metrics.accuracy_score(y_test, y_predicted)

print('Accuracy: %.2f%%' % (score * 100))