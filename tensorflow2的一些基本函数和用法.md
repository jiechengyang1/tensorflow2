# tensorflow2

[toc]

## 1.生成张量

```py
a=np.arange(0,5)
b=tf.convert_to_tensor(a,dtype=tf.int64)
print(a)
print(b)
```

```
[0 1 2 3 4]
tf.Tensor([0 1 2 3 4], shape=(5,), dtype=int64)
```

```pyt
a=tf.zeros([2,3])
b=tf.ones(4)
c=tf.fill([2,2],9)
print(a)
print(b)
print(c)
```

```
tf.Tensor(
[[0. 0. 0.]
 [0. 0. 0.]], shape=(2, 3), dtype=float32)
tf.Tensor([1. 1. 1. 1.], shape=(4,), dtype=float32)
tf.Tensor(
[[9 9]
 [9 9]], shape=(2, 2), dtype=int32)
```

#### tf.random

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220520194734335.png" alt="image-20220520194734335" style="zoom: 67%;" />

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220520194703197.png" alt="image-20220520194703197" style="zoom:67%;" />

##### 不规则张量

如果张量的某个轴上的元素个数可变，则称为“不规则”张量。对于不规则数据，请使用`tf.ragged.RaggedTensor`

## 2.常用函数

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220520194633525.png" alt="image-20220520194633525" style="zoom:67%;" />

#### tf.reduce_mean平均值  求和tf.reduce_sum

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220520194845840.png" alt="image-20220520194845840" style="zoom:67%;" />

#### tf.Variable---变量

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220520195027211.png" alt="image-20220520195027211" style="zoom:67%;" />

#### tf......数学运算

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220520195046166.png" alt="image-20220520195046166" style="zoom:67%;" />

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220520195100022.png" alt="image-20220520195100022" style="zoom:67%;" />

张量的维度应该相同 

![image-20220520195147476](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220520195147476.png)

#### tf.data.Dataset.from_tensor_slices((特征，标签))将特征和标签对应起来

```py
features=tf.constant([12,23,10,17])
labels=tf.constant([0,1,1,0])
dataset=tf.data.Dataset.from_tensor_slices((features,labels))
print(dataset)
for e in dataset:
    print(e)
```

```
<TensorSliceDataset shapes: ((), ()), types: (tf.int32, tf.int32)>
(<tf.Tensor: shape=(), dtype=int32, numpy=12>, <tf.Tensor: shape=(), dtype=int32, numpy=0>)
(<tf.Tensor: shape=(), dtype=int32, numpy=23>, <tf.Tensor: shape=(), dtype=int32, numpy=1>)
(<tf.Tensor: shape=(), dtype=int32, numpy=10>, <tf.Tensor: shape=(), dtype=int32, numpy=1>)
(<tf.Tensor: shape=(), dtype=int32, numpy=17>, <tf.Tensor: shape=(), dtype=int32, numpy=0>)
```

from_tensor_slices函数使输入特征和标签值一一对应。（把数据集分批次，每个批次batch组数据）

train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

#### 求导

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220520195659266.png" alt="image-20220520195659266" style="zoom:67%;" />

#### tf.one_hot---one-hot函数

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220520195843393.png" alt="image-20220520195843393" style="zoom:67%;" />

#### tf.nn.softmax激活函数

![image-20220520200015311](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220520200015311.png)

```py
y=tf.constant([1.01,2.01,-0.66])
y_pro=tf.nn.softmax(y)
print(y_pro)
结果
tf.Tensor([0.25598174 0.6958304  0.0481878 ], shape=(3,), dtype=float32)
```

#### assign_sub

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220520200358594.png" alt="image-20220520200358594" style="zoom:67%;" />

#### tf.argmax求张量方向的最大值

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220520200435460.png" alt="image-20220520200435460" style="zoom:67%;" />

test = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]])
print("test:\n", test)
print("每一列的最大值的索引：", tf.argmax(test, axis=0))  # 返回每一列最大值的**索引**
print("每一行的最大值的索引", tf.argmax(test, axis=1))  # 返回每一行最大值的**索引**

```
test:
 [[1 2 3]
 [2 3 4]
 [5 4 3]
 [8 7 2]]
每一列的最大值的索引： tf.Tensor([3 3 1], shape=(3,), dtype=int64)
每一行的最大值的索引 tf.Tensor([2 2 0 0], shape=(4,), dtype=int64)
```

鸢尾花的实验

![image-20220520200909209](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220520200909209.png)

## 3.优化网络的基础知识

![image-20220521142944647](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220521142944647.png)

#### tf.where:条件语句类似三元表达式

条件语句，a，b

类似三元表达式(a>b)?a:b

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220521143137110.png" alt="image-20220521143137110" style="zoom:67%;" />

#### np.vstack---将数组纵向叠加



<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220521143430170.png" alt="image-20220521143430170" style="zoom:50%;" />

#### np.ngrid[]生成不同维度的向量

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220521143605963.png" alt="image-20220521143605963" style="zoom:50%;" />

比如这里x,y=np.mgrid[1:3:1,2:4:0.5]

按道理x应该是[1,2]

​			y应该是[2,2.5,3,3.5]

但是mgrid会根据二者的维度建立向量。x维度为行

y维度为列，则为2*4的矩阵

### 神经网络的复杂度

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220521144602356.png" alt="image-20220521144602356" style="zoom: 67%;" />

一层隐层+输出层为两层NN

W1（4，3）b1（4，1）

W2（2，4）b2（2，1）

### 激活函数

#### sigmoid函数

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220521145005406.png" alt="image-20220521145005406" style="zoom:67%;" />

sigmoid在深度神经网络导致梯度消失

因为sigmoid函数的导数在0-0.25之间，，深层神经网络多层链式求导会有多个导数值相乘

而导数值在（0，0.25）之间，多个相乘导致数据越来越小，趋向0，导致梯度消失

#### tanh函数

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220521145217533.png" alt="image-20220521145217533" style="zoom:67%;" />

#### Relu函数

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220521145239608.png" alt="image-20220521145239608" style="zoom:67%;" />

#### Leakey Relu--对Relu的优化

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220521145359214.png" alt="image-20220521145359214" style="zoom:50%;" />

首选relu函数激活，学习率选小的，将输入标准特征化

### 损失函数

三种：均方误差

​			自定义

​			交叉熵

![image-20220521145553445](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220521145553445.png)

#### 均方误差mse----tf.reduce_mean(tf.square(y_-y))

![image-20220521145617058](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220521145617058.png)

预测值和标记值差的平方和求均值

#### tf.reduce_sum---损失函数求和

#### tf.losses.categorical_crossentropy(y_,y)交叉熵

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220521150719269.png" alt="image-20220521150719269" style="zoom:67%;" />

#### tf.nn.softmax_cross_entropy_with_logits----先经过softmax在经过交叉熵

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220521151204272.png" alt="image-20220521151204272" style="zoom:67%;" />

### 过拟合vs欠拟合--

学渣---  -学霸----书呆子

欠拟合--刚刚好--过拟合

![image-20220521151410036](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220521151410036.png)

#### 正则化--对w使用

##### tf.nn.l1----L1正则化

##### tf.nn.l2----L2正则化

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220521151541154.png" alt="image-20220521151541154" style="zoom:67%;" />

## 4.神经网络的参数优化器

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220521155653896.png" alt="image-20220521155653896" style="zoom:67%;" />

### 随机梯度下降SGD

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220521155945189.png" alt="image-20220521155945189" style="zoom:67%;" />

这里的gt指的是对loss损失函数对参数w的导数

随机梯度下降是没有使用优化器的，实验结果如下

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220521160317949.png" alt="image-20220521160317949" style="zoom:50%;" />

### SGDM（momentum的SGD）也就是----动量梯度下降法（Gradient descent with Momentum）

在sgd的基础上增加momentum一阶动量

β是超参数，一般取0.9，，此时β比导数值更重要

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220521160612216.png" alt="image-20220521160612216" style="zoom:67%;" />

如下图所示

![image-20220521160812075](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220521160812075.png)

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220521160931243.png" alt="image-20220521160931243" style="zoom:50%;" />

这里发现在相同学习率的情况下SGDM时间居然更长？

是没有效果吗，并不是，此时我们可以加大学习率，好吧还是更长，麻了。。。。

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220521161421688.png" alt="image-20220521161421688" style="zoom:67%;" />

### Adagrad优化器-

在SDG的基础上加入二阶动量

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220521161839578.png" alt="image-20220521161839578" style="zoom:50%;" />

一阶动量是导数（梯度）

二阶动量是导数平方和

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220521161940338.png" alt="image-20220521161940338" style="zoom:67%;" />

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220521162220300.png" alt="image-20220521162220300" style="zoom:50%;" />

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220521162601174.png" alt="image-20220521162601174" style="zoom:50%;" />

### RMSProp优化器

在SGD的基础上增加二阶动量，此时的二阶动量和上面不一样

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220521162333479.png" alt="image-20220521162333479" style="zoom:67%;" />

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220521162454086.png" alt="image-20220521162454086" style="zoom:67%;" />

保持跟之前一样的学习率，，这是学习率过高导致的，可以适当降低学习率

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220521162733181.png" alt="image-20220521162733181" style="zoom:50%;" />

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220521162742064.png" alt="image-20220521162742064" style="zoom:50%;" />

### ♥Adam优化器--结合SGD和RMSProp



<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220521163118215.png" alt="image-20220521163118215" style="zoom:67%;" />

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220521163232064.png" alt="image-20220521163232064" style="zoom:67%;" />

t代表的是经过的batch数，，就是分成多少个batch

也就是上面的t或者global_step

t会不断增加，没迭代一次batch就会增加1

很明显adamyyds

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220521163545063.png" alt="image-20220521163545063" style="zoom:50%;" />

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220521163602676.png" alt="image-20220521163602676" style="zoom:50%;" />