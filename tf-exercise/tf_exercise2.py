#1.同样是对minist数据集进行分类，采用最开始的不使用keras的框架实现一遍
#2.学习下python中的class
import tensorflow as tf
from sklearn import datasets
import keras
import matplotlib.pyplot as plt
import numpy as np
#加载训练数据与测试数据
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

print(x_data[0])  #[5.1,3.5,1.4,0.2]
print(x_data.shape) #(150,4)
#数据集乱序
np.random.seed(116) #设置随机种子
np.random.shuffle(x_data) #重新排序返回一个随机序列 类似洗牌
np.random.seed(116)
np.random.shuffle(y_data)
print(y_data[0])  #1
print(y_data.shape) #(150,)

#划分测试集与训练集
print(type(x_data))  #numpy.ndarray

#划分永不相见的训练集和测试集
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

x_train = tf.cast(x_train,dtype=tf.float32)
x_test = tf.cast(x_test,dtype=tf.float32)
#进行配对操作 利用from_tensor_slices((features,label))
train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)

###########训练部分开始###########
#定义神经网络中所有可训练参数
W1 = tf.Variable(tf.random.truncated_normal([4,3],stddev=0.1,seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3,],stddev=0.1,seed=1))
epoch = 100
loss_all=0
lr = 0.1
train_loss_results=[]
test_acc=[]  #这个 一不小心给放到循环里面了，因此每次进行循环时，都进行了清空，数据无法保存
#嵌套循环迭代
#print("train_db:",train_db)
#train_db本身就是一个列表类型
for epoch in range(epoch):
    for step,(x_train,y_train) in enumerate(train_db):  #对于每个epoch中的每个batch中的样本而言的前向传播
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train,W1)+b1   #得到z
            y = tf.nn.softmax(y)      #经过激活函数 得到a
            y_ = tf.one_hot(y_train,depth=3) #将输出y转化成one-hot向量
            loss = tf.reduce_mean(tf.square(y-y_)) #需要进行一个降维的操作
            loss_all += loss.numpy()  #.numpy()用法存疑
        #num += 1
        #print("Step:{}".format(step))
    #前向传播搭建完成，现在开始进行梯度计算和参数自更新
        #每个batch就要更新一次，因此它应该是在for循环里面鸭
        grads = tape.gradient(loss,[W1,b1])
        W1.assign_sub(lr*grads[0])             #哭了，搞半天梯度下降的方向反了
        b1.assign_sub(lr*grads[1])

    print("Epoch{},loss:{}".format(epoch,loss_all/4))   #输出本epoch次的损失信息
    train_loss_results.append(loss_all/4)
    loss_all =0  #记得清0，为下一次epoch做准备
    #print("Num:{}".format(num))

#Tips:tab键进行缩进，tab+shift取消缩进
###########显示当前模型效果############
    total_correct=0
    total_number=0
    for x_test,y_test in test_db:  #test_db是一个特征标签对
        #利用当前的参数值计算出预测值

        y = tf.matmul(x_test,W1)+b1
        y = tf.nn.softmax(y)
        pred = tf.argmax(y,axis=1)
        pred = tf.cast(pred,dtype=y_test.dtype)

        #计算总的正确预测数
        correct = tf.cast(tf.equal(y_test,pred),dtype=tf.int32)
        correct = tf.reduce_sum(correct)  #计算每个batch中的总正确数
        total_correct += int(correct)  #将所有batch中正确的数目进行相加

        #计算总的样本个数
        total_number += x_test.shape[0]

        #计算准确率
    acc = total_correct/total_number
    test_acc.append(acc)
    print("Test acc:",acc)
    print("******************************************")

#绘制loss曲线
plt.title("Loss Function Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(train_loss_results,label="$Loss$")
plt.legend()    #
plt.show()
#绘制acc曲线
plt.title("Acc Function Curve")
plt.xlabel("Epoch")
plt.ylabel("Acc")
plt.plot(test_acc,label="$Accuracy$")
plt.legend()
plt.show()






