## 谷歌AI开源张量计算库TensorNetwork，计算速度暴涨100倍

[新智元](javascript:void(0);) *昨天*

![img](https://mmbiz.qpic.cn/mmbiz_jpg/UicQ7HgWiaUb2BROlqmbtIXdKtwVYHCOlqHL4mmv1gFje0gjibYmicAkibtStWsZ3RmKfd8v0TZog9jWIsMCDN8Ss4g/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

###    **新智元报道**  

来源：Google AI

编辑：大明、张佳

##### **【新智元导读】**谷歌宣布开源张量计算库TensorNetwork及其API，使用TensorFlow为后端，对GPU处理速度进行优化，与CPU相比，计算加速效果高达100倍。

现代科学领域中，有很多艰巨困难的科学任务，比如开发高温超导体材料、了解空间和时间的本质等，都涉及到处理量子系统的复杂性。这些问题之所以困难，是因为这些**系统中的量子态数量呈指数级增长，使得暴力计算行不通了。**

 

为了解决这个问题，人们利用名为“张量网络”的数据结构，可以专注于与现实问题最为相关的量子态——低能量状态，而忽略其他不相关的状态。张量网络也越来越多地在机器学习中得到应用。



然而，目前在机器学习中应用张量还存在一些困难：比如用于加速硬件的生产级张量网络库尚未在大规模运行张量网络算法中部署，而且，大多数关于张量网络的文献是面向物理学科领域的应用。这也让人们产生一种错误印象，认为需要掌握量子力学的专业知识才能理解张量算法。

 

**本次开源的TensorNetwork使用TensorFlow作为后端，并针对GPU处理进行了优化，与CPU相比，处理速度可以实现100倍的加速。**此前已经介绍了TensorNetwork，包括新的库及其API，并针对非物理学背景的读者对张量网络进行了概述，介绍了张量网络在物理学中的特定应用实例，展示了使用GPU带来的处理速度的显著提升。

 

为什么Tensor Networks有用？从张量的图解表示说起



**张量**是一种多维数组，根据数组元素的顺序按层级分类：例如，普通数是零阶张量（也称为标量），向量可视为一阶张量，矩阵可视为二阶张量等等。低阶张量可以很容易用一个明确的数字数组或数学符号来表示。



不过涉及到高阶时，这种符号法就变得非常麻烦。**使用图解符号对于解决这个问题很有用**，一种方法是简单地绘制一个圆（或其他形状），引出多条线或者说“腿”，腿的数量与张量的阶数相同。在这种表示法中，标量表示为一个圆，矢量有一条腿，矩阵有两条腿等。张量的每条腿也有一个尺寸，就是腿的长短。例如，表示物体通过空间的速度的矢量就是三维的一阶张量。

 

![img](https://mmbiz.qpic.cn/mmbiz_png/UicQ7HgWiaUb2BROlqmbtIXdKtwVYHCOlqhWXRgscDQrRmSb2UDKEROY1hvUcHfHO2wFia0icWhnibvcpDrdrwclthQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

张量的图解表示法

 

**以这种方式表示张量的好处是可以简洁地对数学运算进行编码，**例如，用矩阵乘以向量，获得另一个向量，或者将两个向量相乘，得到一个标量。这些都是所谓“张量收缩”的更一般的概念。

 

![img](https://mmbiz.qpic.cn/mmbiz_png/UicQ7HgWiaUb2BROlqmbtIXdKtwVYHCOlqOKcphGlqIqm82iciaLrmjrZeC7hPicyHp87vGTjNLOG1d9c2oRVqKNHCA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

张量收缩的图解表示法。图中所示为矢量和矩阵乘法以及矩阵迹线（即矩阵的对角元素的总和）

 

以下是张量网络的简单示例。张量网络是对几个张量收缩，形成新张量的模式进行编码的图形化表示。构成新张量的每个张量具有各自的阶数，图上表示为腿的数量。互相连接的腿，在图中形成边，表示张量的收缩，而剩余的悬在外面的腿的数量就是生成的新张量的阶数。

 

![img](https://mmbiz.qpic.cn/mmbiz_png/UicQ7HgWiaUb2BROlqmbtIXdKtwVYHCOlqdpnJkkibsNxz7xqJnyOQAqLNJL5jCiag29ANS8v1lyAmF2rbFzXvjN5A/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

左：四个矩阵乘积的表示，即tr（ABCD），它是一个标量，图中没有腿。右：三个三阶张量收缩，最终有三条腿悬在外面，即产生一个新的三阶张量。



虽然这些例子非常简单，但张量网络通常代表以各种方式收缩的数百个张量。用传统的符号来描述这样一件事是很难理解的，这就是Roger Penrose在1971年发明图解符号（diagrammatic notation）的原因。

 

**张量网络在实践中的应用**



想象一组黑白图像，每个图像都可以看作是一个n个像素值的列表。单个图像的单个像素可以被一个one-hot编码为二维矢量，通过将这些像素编码结合在一起，我们可以对整个图像进行2N维的one-hot编码。我们可以将这个高维向量重塑成一个order-*N*张量，然后将图像集中的所有张量相加，得到一个总张量T*i1*,*i2*,*...*,*iN*集合。

 

这听起来是一件非常浪费的事：用这种方式编码大约50像素的图像将占内存许多PB的空间。这就该用到张量网络了。与其直接存储或操纵张量 *T*，不如将 *T*表示为张量网络形状中许多较小组分张量的收缩。结果证明效率更高。例如，流行的矩阵积态（MPS）网络将把 *T*写成N个更小的张量，这样参数的总数在N中只是线性的，而不是指数的。

 

![img](https://mmbiz.qpic.cn/mmbiz_png/UicQ7HgWiaUb2BROlqmbtIXdKtwVYHCOlq27KsVickjxcFmoict6eqt27cxSbcqhINJT4avxNiciap3RhTGtX6PoPIvA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

 在矩阵积态张量网络中，高阶张量T用许多低阶张量表示。

 

不明显的是，大张量网络可以被有效地创建或操作，同时始终避免占用大量内存。但事实证明，这在许多情况下是可能的，这就是为什么张量网络在量子物理学和现在的机器学习中被广泛使用的原因。



谷歌AI的研究人员Stoudenmire和Schwab使用刚才描述的编码来建立一个图像分类模型，展示了张量网络的新用途。TensorNetwork库的设计就是为了方便这种工作，我们第一篇论文（https://arxiv.org/pdf/1905.01330.pdf）就描述了该库如何用于一般的张量网络操作。

 

**性能实例分析：计算速度提升100倍**

 

**张量网络是张量网络算法的通用库，对物理学家也有一定的帮助。**量子态的近似是物理中张量网络的一个典型用例，非常适合用来说明张量网络库的功能。在第二篇论文（https://arxiv.org/pdf/1905.01331.pdf）中，我们描述了一种tree tensor network（TTN）算法，用于估算周期性量子自旋链（1D）或薄环面上的晶格模型（2D）的基态，并用张量网络实现了该算法。在使用GPU和TensorNetwork库时，**我们比较了CPU和GPU的使用情况，并观察到计算速度显著提高，高达100倍。**

 

![img](https://mmbiz.qpic.cn/mmbiz_png/UicQ7HgWiaUb2BROlqmbtIXdKtwVYHCOlq5leQsQ0LTpeibQ8gDVZSO3vwxib5PAQBmsJqPPGjTAsuZEvyicZNt5BwA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

计算时间作为键维数的函数，χ。 键合维度决定了张量网络的组成张量的大小。 更大的键维度意味着更强大的张量网络，但需要更多的计算资源来操纵。

 

**未来方向：时间序列分析和量子电路模拟**

 

我们计划用一系列的论文来说明张量网络在实际应用中的强大之处，这是第一篇。在下一篇论文中，我们将使用TensorNetwork对MNIST和Fashion-MNIST数据集中的图像进行分类。



未来的计划包括**机器学习方面的时间序列分析和物理方面的量子电路模拟。**通过开源社区，我们会经常为TensorNetwork添加新功能。我们希望TensorNetwork将成为物理学家和机器学习实践者的宝贵工具。

**参考链接：**

https://ai.googleblog.com/2019/06/introducing-tensornetwork-open-source.html



**论文资源：**

https://arxiv.org/pdf/1905.01330.pdf

https://arxiv.org/pdf/1905.01331.pdf



**GitHub资源：**

https://github.com/google/TensorNetwork

**新智元春季招聘开启，****一起弄潮AI之巅！**

**岗位详情请戳：**

[![解决AI技术落地难题，“解耦”是关键 (3).png](https://mmbiz.qpic.cn/mmbiz_png/UicQ7HgWiaUb2M4h9tkuarGklADG9cjGMsf8bicLRzt5cibWevRjGhqg5Nr6MNwCbbSmV2WE1PdyLqytGrKJms8R0w/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)](http://mp.weixin.qq.com/s?__biz=MzI3MTA0MTk1MA==&mid=2652040487&idx=5&sn=4d39d27bf481f4651c17aa58f8e08436&chksm=f12199d6c65610c006f6640fccf6c28ace29138a132b8f6b60daa53329894dd006aaa751ea15&scene=21#wechat_redirect)



**【加入社群】**



新智元AI技术+产业社群招募中，欢迎对AI技术+产业落地感兴趣的同学，加小助手微信号：aiera2015_2   入群;通过审核后我们将邀请进群，加入社群后务必修改群备注（姓名 - 公司 - 职位;专业群审核较严，敬请谅解）。

![img](https://mmbiz.qpic.cn/mmbiz_gif/UicQ7HgWiaUb1KTwONTiaO3FZYUSGxl8ibiaHPViaYfsE4hOOOHrmyQ7r5CwkByn6oHdGmwBA6Q1I6r4eCn9gVhJQ3nA/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)









微信扫一扫
关注该公众号