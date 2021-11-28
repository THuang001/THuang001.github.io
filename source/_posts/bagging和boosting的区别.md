  bagging和boosting的区别

---

title: 面试常问问题复习(二)
date: 2019-07-02 23:35:58
tags: 面试
categories: 算法

---



Bagging和Boosting都是将已有的分类或回归算法通过一定方式组合起来，形成一个性能更加强大的分类器，更准确的说这是一种分类算法的组装方法。即将弱分类器组装成强分类器的方法。

       首先介绍Bootstraping，即自助法：它是一种有放回的抽样方法（可能抽到重复的样本）。

1. ## **Bagging (bootstrap aggregating)**

  Bagging即套袋法，其算法过程如下：

从原始样本集中抽取训练集。每轮从原始样本集中使用Bootstraping的方法抽取n个训练样本（在训练集中，有些样本可能被多次抽取到，而有些样本可能一次都没有被抽中）。共进行k轮抽取，得到k个训练集。（k个训练集之间是相互独立的）

每次使用一个训练集得到一个模型，k个训练集共得到k个模型。（注：这里并没有具体的分类算法或回归方法，我们可以根据具体问题采用不同的分类或回归方法，如决策树、感知器等）

对分类问题：将上步得到的k个模型采用投票的方式得到分类结果；对回归问题，计算上述模型的均值作为最后的结果。（所有模型的重要性相同）

2. ## **Boosting**
  
  ​     其主要思想是将弱分类器组装成一个强分类器。在PAC（概率近似正确）学习框架下，则一定可以将弱分类器组装成一个强分类器。

关于Boosting的两个核心问题：

### **2.1 在每一轮如何改变训练数据的权值或概率分布？**

​       通过提高那些在前一轮被弱分类器分错样例的权值，减小前一轮分对样例的权值，来使得分类器对误分的数据有较好的效果。

### 2.2 通过什么方式来组合弱分类器？

​       通过加法模型将弱分类器进行线性组合，比如AdaBoost通过加权多数表决的方式，即增大错误率小的分类器的权值，同时减小错误率较大的分类器的权值。

而提升树通过拟合残差的方式逐步减小残差，将每一步生成的模型叠加得到最终模型。

3. ## Bagging，Boosting二者之间的区别

  Bagging和Boosting的区别：

1）样本选择上：

Bagging：训练集是在原始集中有放回选取的，从原始集中选出的各轮训练集之间是独立的。

Boosting：每一轮的训练集不变，只是训练集中每个样例在分类器中的权重发生变化。而权值是根据上一轮的分类结果进行调整。

2）样例权重：

Bagging：使用均匀取样，每个样例的权重相等

Boosting：根据错误率不断调整样例的权值，错误率越大则权重越大。

3）预测函数：

Bagging：所有预测函数的权重相等。

Boosting：每个弱分类器都有相应的权重，对于分类误差小的分类器会有更大的权重。

4）并行计算：

Bagging：各个预测函数可以并行生成

Boosting：各个预测函数只能顺序生成，因为后一个模型参数需要前一轮模型的结果。

4. # 为什么说bagging是减少variance，而boosting是减少bias?

Bagging对样本重采样，对每一重采样得到的子样本集训练一个模型，最后取平均。由于子样本集的相似性以及使用的是同种模型，因此各模型有近似相等的bias和variance（事实上，各模型的分布也近似相同，但不独立）。由于![[公式]](https://www.zhihu.com/equation?tex=E%5B%5Cfrac%7B%5Csum+X_i%7D%7Bn%7D%5D%3DE%5BX_i%5D)，所以bagging后的bias和单个子模型的接近，一般来说不能显著降低bias。另一方面，若各子模型独立，则有![[公式]](https://www.zhihu.com/equation?tex=Var%28%5Cfrac%7B%5Csum+X_i%7D%7Bn%7D%29%3D%5Cfrac%7BVar%28X_i%29%7D%7Bn%7D)，此时可以显著降低variance。若各子模型完全相同，则![[公式]](https://www.zhihu.com/equation?tex=Var%28%5Cfrac%7B%5Csum+X_i%7D%7Bn%7D%29%3DVar%28X_i%29)

，此时不会降低variance。bagging方法得到的各子模型是有一定相关性的，属于上面两个极端状况的中间态，因此可以一定程度降低variance。为了进一步降低variance，Random forest通过随机选取变量子集做拟合的方式de-correlated了各子模型（树），使得variance进一步降低。

（用公式可以一目了然：设有i.d.的n个随机变量，方差记为![[公式]](https://www.zhihu.com/equation?tex=%5Csigma%5E2)，两两变量之间的相关性为![[公式]](https://www.zhihu.com/equation?tex=%5Crho)，则![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Csum+X_i%7D%7Bn%7D)的方差为![[公式]](https://www.zhihu.com/equation?tex=%5Crho%2A%5Csigma%5E2%2B%281-%5Crho%29%2A%5Csigma%5E2%2Fn)，bagging降低的是第二项，random forest是同时降低两项。详见ESL p588公式15.1）

boosting从优化角度来看，是用forward-stagewise这种贪心法去最小化损失函数![[公式]](https://www.zhihu.com/equation?tex=L%28y%2C+%5Csum_i+a_i+f_i%28x%29%29)。例如，常见的AdaBoost即等价于用这种方法最小化exponential loss：![[公式]](https://www.zhihu.com/equation?tex=L%28y%2Cf%28x%29%29%3Dexp%28-yf%28x%29%29)。所谓forward-stagewise，就是在迭代的第n步，求解新的子模型f(x)及步长a（或者叫组合系数），来最小化![[公式]](https://www.zhihu.com/equation?tex=L%28y%2Cf_%7Bn-1%7D%28x%29%2Baf%28x%29%29)，这里![[公式]](https://www.zhihu.com/equation?tex=f_%7Bn-1%7D%28x%29)是前n-1步得到的子模型的和。因此boosting是在sequential地最小化损失函数，其bias自然逐步下降。但由于是采取这种sequential、adaptive的策略，各子模型之间是强相关的，于是子模型之和并不能显著降低variance。所以说boosting主要还是靠降低bias来提升预测精度。

