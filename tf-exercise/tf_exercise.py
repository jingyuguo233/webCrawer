#导入所需的数据包
import tensorflow as tf
from sklearn import datasets
import keras
import matplotlib.pyplot as plt
import numpy as np
#导入数据集
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
#对数据进行归一化等预处理
x_train = x_train/255.0
x_test = x_test/255.0
#观察数据集 0.维度 1.查看原始图像
#每个图片的尺寸是28*28，训练集共有60000条数据,测试集有10000条数据
"""
print(x_train.shape)
print(y_train[0])
print(x_train[0])
print(y_test.shape)
plt.imshow(x_train[0])
plt.show()
"""
#搭建模型
model = tf.keras.models.Sequential(
    [tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(20,activation=tf.nn.relu),
    tf.keras.layers.Dense(15,activation=tf.nn.relu),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)]
)

#在compile中设置优化算法、损失函数和精确度显示形式
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy']
              )

#fit中设置训练的epoch、batch及数据集的划分
model.fit(x_train,y_train,batch_size=32,epochs=5,validation_data=(x_test,y_test),validation_freq=20)

#显示模型的汇总信息
model.summary()
