---

title: FFM模型
date: 2019-07-02 23:35:58
tags: 论文
categories: 模型

---

## 1. 背景

FFM（Field-aware Factorization Machine）最初的概念来自Yu-Chin Juan（阮毓钦，毕业于中国台湾大学，现在美国Criteo工作）与其比赛队员，是他们借鉴了来自Michael Jahrer的论文[[1\]](https://kaggle2.blob.core.windows.net/competitions/kddcup2012/2748/media/Opera.pdf)中的field概念提出了FM的升级版模型。通过引入field的概念，FFM把相同性质的特征归于同一个field。

## 2. FFM原理推导

考虑下面的数据集：

| Clicked? | Publisher(P) | Advertiser(A) | Gender(G) |
| :------- | :----------- | :------------ | :-------- |
| 1        | EPSN         | Nike          | Male      |
| 0        | NBC          | Adidas        | Female    |

对于第一条数据来说，FM模型的二次项为：**w**𝐸𝑃𝑆𝑁⋅**𝐰**𝑁𝑖𝑘𝑒+**𝐰**𝐸𝑃𝑆𝑁⋅**𝐰**𝑀𝑎𝑙𝑒+**𝐰**𝑁𝑖𝑘𝑒⋅**𝐰**𝑀𝑎𝑙𝑒。（这里只是把上面的v符合改成了w）每个特征只用一个隐向量来学习和其它特征的潜在影响。对于上面的例子中，Nike是广告主，Male是用户的性别，描述（EPSN，Nike）和（EPSN，Male）特征组合，FM模型都用同一个**𝐰**𝐸𝑆𝑃𝑁，而实际上，ESPN作为广告商，其对广告主和用户性别的潜在影响可能是不同的。

因此，Yu-Chin Juan借鉴Michael Jahrer的论文（Ensemble of collaborative filtering and feature engineered models for click through rate prediction），将field概念引入FM模型。

field是什么呢？即相同性质的特征放在一个field。比如EPSN、NBC都是属于广告商field的，Nike、Adidas都是属于广告主field，Male、Female都是属于性别field的。简单的说，同一个类别特征进行one-hot编码后生成的数值特征都可以放在同一个field中，比如最开始的例子中Day=26/11/15 Day=19/2/15可以放于同一个field中。如果是数值特征而非类别，可以直接作为一个field。

引入了field后，对于刚才的例子来说，二次项变为：

{% raw %}​

$\underbrace{{\bf w}_{EPSN, A} \cdot {\bf w}_{Nike, P}}_{P \times A} + \underbrace{{\bf w}_{EPSN, G} \cdot {\bf w}_{Male,P}}_{P \times G} + \underbrace{{{\bf w}_{Nike, G} \cdot {\bf w}_{Male,A}}}_{A \times G}$

{% endraw %}​

- 对于特征组合（EPSN，Nike）来说，其隐向量采用的是**𝐰**𝐸𝑃𝑆𝑁,𝐴和**𝐰**𝑁𝑖𝑘𝑒,𝑃，对于**𝐰**𝐸𝑃𝑆𝑁,𝐴这是因为Nike属于广告主（Advertiser）的field，而第二项**𝐰**𝑁𝑖𝑘𝑒,𝑃则是EPSN是广告商（Publisher）的field。
- 再举个例子，对于特征组合（EPSN，Male）来说，**𝐰**𝐸𝑃𝑆𝑁,𝐺 是因为Male是用户性别(Gender)的field，而第二项**𝐰**𝑀𝑎𝑙𝑒,𝑃是因为EPSN是广告商（Publisher）的field。

下面的图来自criteo，很好的表示了三个模型的区别

> For Poly2, a dedicated weight is learned for each feature pair: 
>
> ![img](https://www.hrwhisper.me/wp-content/uploads/2018/07/poly2-model-example.png)
>
> For FMs, each feature has one latent vector, which is used to interact with any other latent vectors:
>
> ![img](https://www.hrwhisper.me/wp-content/uploads/2018/07/fm-model-example.png)
>
> For FFMs, each feature has several latent vectors, one of them is used depending on the field of the other feature:
>
> ![img](https://www.hrwhisper.me/wp-content/uploads/2018/07/ffm-model-example.png)





## 3. FFM模型学习

### 3.1 FFM 数学公式

假设样本的 n 个特征属于 f 个field，那么FFM的二次项有 $nf$个隐向量。而在FM模型中，每一维特征的隐向量只有一个。FM可以看作FFM的特例，是把所有特征都归属到一个field时的FFM模型。根据FFM的field敏感特性，可以导出其模型方程。

$y(\mathbf{x}) = w_0 + \sum_{i=1}^d w_i x_i + \sum_{i=1}^d \sum_{j=i+1}^d (w_{i, f_j} \cdot w_{j, f_i}) x_i x_j  \tag{3-0}$

其中，$f_j$是第j个特征所属的field。如果隐向量的长度为k，那么FFM的二次参数有$nfk$个，远多于FM模型的nk个。此外，由于隐向量与field相关，FFM二次项并不能够化简，其复杂度为$O(kn^2)$。

### 3.2 FFM 模型学习

为了方便推导，这里省略FFM的一次项和常数项，公式为：

$\phi(\mathbf{w}, \mathbf{x}) =\sum_{a=1}^d \sum_{b=a+1}^d ( w_{a, f_b} \cdot w_{b, f_a}) x_a x_b\tag{3-1}$

FFM模型使用logistic loss作为损失函数，并加上L2正则项：

$\mathcal{L} = \sum_{i=1}^N\log\left(1 + \exp(-y_i\phi({\bf w}, {\bf x_i}))\right) + \frac{\lambda}{2} |\!|{\bf w}|\!|^2 \tag{3-2}$

采用随机梯度下降来（SGD）来优化损失函数，因此，损失函数只采用单个样本的损失：

$\mathcal{L} =\log\left(1 + \exp(-y_i\phi({\bf w}, {\bf x}))\right) + \frac{\lambda}{2} |\!|{\bf w}|\!|^2 \tag{3-3}$

对于每次迭代，选取一条数据(**𝐱**,𝑦)，然后让L对**𝐰**𝑎,𝑓𝑏和**𝐰**𝑏,𝑓𝑎求偏导（注意，采用SGD上面的求和项就去掉了，只采用单个样本的损失），得：

{% raw %}

$\begin{align}  g_{a,f_b} \equiv \frac{\partial \mathcal{L}}{\partial w_{a,f_b}} = \kappa\cdot w_{b, f_a} x_a x_b + \lambda w_{a,f_b}^2 \tag{3-4} \\  g_{b,f_a} \equiv \frac{\partial \mathcal{L}}{\partial w_{b,f_a}} = \kappa\cdot w_{a, f_b} x_a x_b + \lambda w_{b,f_a}^2 \tag{3-5}\\  其中, \kappa = \frac{-y}{1+\exp(y\phi({\bf w,x}))}  \tag{3-6}\end{align}$

{% endraw %}

在具体的实现中，这里有两个trick，

第一个trick是梯度的分步计算。

{% raw %}

$\mathcal{L} = \mathcal{L} _{err} + \mathcal{L} _{reg} = \log\left(1 + \exp(-y_i\phi({\bf w}, {\bf x}))\right) + \frac{\lambda}{2} |\!|{\bf w}|\!|^2\\  \frac{\partial\mathcal{L}}{\partial\mathbf{w}} = \frac{\partial\mathcal{L}_{err}}{\partial\phi}\cdot \frac{\partial\phi}{\partial\mathbf{w}} + \frac{\partial\mathcal{L}_{reg}}{\partial\mathbf{w}}\tag{3-7}$

{% endraw %}

注意到$\frac{\partial\mathcal{L}_{err}}{\partial\phi}$和参数无关，每次更新模型时，只需要计算一次，之后直接调用结果即可。对于总共有𝑑𝑓𝑘个模型参数的计算来说，使用这种方式能极大提升运算效率。

第二个trick是FFM的学习率是随迭代次数变化的，具体的是采用[AdaGrad](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#AdaGrad)算法，这里进行简单的介绍。

Adagrad算法能够在训练中自动的调整学习率，**对于稀疏的参数增加学习率，而稠密的参数则降低学习率。因此，Adagrad非常适合处理稀疏数据。**

设𝑔𝑡,𝑗为第t轮第j个参数的梯度，则SGD和采用Adagrad的参数更新公式分别如下：

{% raw %}

$\begin{align*}  SGD: \ & w_{t+1,j} = w_{t,j} -\eta \cdot g_{t,j} \tag{3-8}\\  Adagrad: \ & w_{t+1,j} = w_{t,j} – \frac{\eta}{\sqrt{G_{t,jj}+ \epsilon}} \cdot g_{t,j}  \tag{3-9}\end{align*}$

{% endraw %}

可以看出，Adagrad在学习率𝜂上还除以一项$\sqrt{G_{t,jj}+ \epsilon}$，这是什么意思呢？𝜖为平滑项，防止分母为0，$G_{t,jj} = \sum_{\iota=1}^tg_{\iota, jj}^2$即𝐺𝑡,𝑗𝑗为对角矩阵，每个对角线位置𝑗,𝑗的值为参数𝑤𝑗每一轮的平方和，可以看出，随着迭代的进行，每个参数的历史梯度累加到一起，使得每个参数的学习率逐渐减小。

因此，用3-4、3-5计算完梯度后，下一步就是更新分母的对角矩阵。

{% raw %}

$\begin{align}  G_{a,f_b} \leftarrow G_{a,f_b} + (g_{a,f_b})^2 \tag{3-10}\\  G_{b,f_a} \leftarrow G_{b,f_a} + (g_{b,f_a})^2 \tag{3-11}  \end{align}$

{% endraw %}

最后，更新模型参数：

{% raw %}

$\begin{align}  w_{a,f_b} &\leftarrow w_{a,f_b} – \frac{\eta}{\sqrt{G_{a,f_b}+ 1}}g_{a,f_b} \tag{3-12}\\  w_{b,f_a} &\leftarrow w_{b,f_a} – \frac{\eta}{\sqrt{G_{b,f_a}+ 1}}g_{b,f_a} \tag{3-13}  \end{align}$

{% endraw %}

这就是论文中算法1描述的过程：

![img](https://www.hrwhisper.me/wp-content/uploads/2018/07/ffm-model-training.png)

![img](https://awps-assets.meituan.net/mit-x/blog-images-bundle-2016/0ba057eb.png)

参考 \($ Algorithm\; 1$ \), 下面简单解释一下FFM的SGD优化过程。 算法的输入 $( tr )、(va)、( pa ) $分别是训练样本集、验证样本集和训练参数设置。

1. 根据样本特征数量$（ tr.n)$、field的个数$( tr.m )$和训练参数$( pa)$，生成初始化模型，即随机生成模型的参数；

2. 如果归一化参数 $( pa.norm )$ 为真，计算训练和验证样本的归一化系数，样本 $( i ) $的归一化系数为

   $R[i] = \frac{1}{\| \mathbf{X}[i] \|}$

3. 对每一轮迭代，如果随机更新参数 \( pa.rand \) 为真，随机打乱训练样本的顺序；

4. 对每一个训练样本，执行如下操作

   - 计算每一个样本的FFM项，$\phi $；
   - 计算每一个样本的训练误差，如算法所示，这里采用的是交叉熵损失函数$\log ( 1 + e\phi )$；
   - 利用单个样本的损失函数计算梯度$ g_\Phi $，再根据梯度更新模型参数；

5. 对每一个验证样本，计算样本的FFM输出，计算验证误差；

6. 重复步骤3~5，直到迭代结束或验证误差达到最小。

### 3.3 实现的trick

本小节主要摘录美团点评的内容。

除了上面提到的梯度分步计算和自适应学习率两个trick外，还有：

> 1. OpenMP多核并行计算。OpenMP是用于共享内存并行系统的多处理器程序设计的编译方案，便于移植和多核扩展[[1\]](http://openmp.org/wp/openmp-specifications/)。FFM的源码采用了OpenMP的API，对参数训练过程SGD进行了多线程扩展，支持多线程编译。因此，OpenMP技术极大地提高了FFM的训练效率和多核CPU的利用率。在训练模型时，输入的训练参数ns_threads指定了线程数量，一般设定为CPU的核心数，便于完全利用CPU资源。
> 2. SSE3指令并行编程。SSE3全称为数据流单指令多数据扩展指令集3，是CPU对数据层并行的关键指令，主要用于多媒体和游戏的应用程序中[[2\]](http://blog.csdn.net/gengshenghong/article/details/7008704)。SSE3指令采用128位的寄存器，同时操作4个单精度浮点数或整数。SSE3指令的功能非常类似于向量运算。例如，a和b采用SSE3指令相加（a和b分别包含4个数据），其功能是a种的4个元素与b中4个元素对应相加，得到4个相加后的值。采用SSE3指令后，向量运算的速度更加快捷，这对包含大量向量运算的FFM模型是非常有利的。
>
> 除了上面的技巧之外，FFM的实现中还有很多调优技巧需要探索。例如，代码是按field和特征的编号申请参数空间的，如果选取了非连续或过大的编号，就会造成大量的内存浪费；在每个样本中加入值为1的新特征，相当于引入了因子化的一次项，避免了缺少一次项带来的模型偏差等。



## 4. 适用范围和使用技巧

在FFM原论文中，作者指出，FFM模型对于one-hot后类别特征十分有效，但是如果数据不够稀疏，可能相比其它模型提升没有稀疏的时候那么大，此外，对于数值型的数据效果不是特别的好。

在Github上有FFM的[开源实现](https://github.com/guestwalk/libffm)，要使用FFM模型，特征需要转化为“**field_id:feature_id:value**”格式，相比LibSVM的格式多了field_id，即特征所属的field的编号，feature_id是特征编号，value为特征的值。

此外，美团点评的文章中，提到了训练FFM时的一些注意事项：

> 第一，样本归一化。FFM默认是进行样本数据的归一化的 。若不进行归一化，很容易造成数据inf溢出，进而引起梯度计算的nan错误。因此，样本层面的数据是推荐进行归一化的。
>
> 第二，特征归一化。CTR/CVR模型采用了多种类型的源特征，包括数值型和categorical类型等。但是，categorical类编码后的特征取值只有0或1，较大的数值型特征会造成样本归一化后categorical类生成特征的值非常小，没有区分性。例如，一条用户-商品记录，用户为“男”性，商品的销量是5000个（假设其它特征的值为零），那么归一化后特征“sex=male”（性别为男）的值略小于0.0002，而“volume”（销量）的值近似为1。特征“sex=male”在这个样本中的作用几乎可以忽略不计，这是相当不合理的。因此，将源数值型特征的值归一化到[0,1]是非常必要的。
>
> 第三，省略零值特征。从FFM模型的表达式(3-1)可以看出，零值特征对模型完全没有贡献。包含零值特征的一次项和组合项均为零，对于训练模型参数或者目标值预估是没有作用的。因此，可以省去零值特征，提高FFM模型训练和预测的速度，这也是稀疏样本采用FFM的显著优势。



在DSP的场景中，FFM主要用来预估站内的CTR和CVR，即一个用户对一个商品的潜在点击率和点击后的转化率。

CTR和CVR预估模型都是在线下训练，然后用于线上预测。两个模型采用的特征大同小异，主要有三类：用户相关的特征、商品相关的特征、以及用户-商品匹配特征。用户相关的特征包括年龄、性别、职业、兴趣、品类偏好、浏览/购买品类等基本信息，以及用户近期点击量、购买量、消费额等统计信息。商品相关的特征包括所属品类、销量、价格、评分、历史CTR/CVR等信息。用户-商品匹配特征主要有浏览/购买品类匹配、浏览/购买商家匹配、兴趣偏好匹配等几个维度。

为了使用FFM方法，所有的特征必须转换成“field_id:feat_id:value”格式，field_id代表特征所属field的编号，feat_id是特征编号，value是特征的值。数值型的特征比较容易处理，只需分配单独的field编号，如用户评论得分、商品的历史CTR/CVR等。categorical特征需要经过One-Hot编码成数值型，编码产生的所有特征同属于一个field，而特征的值只能是0或1，如用户的性别、年龄段，商品的品类id等。除此之外，还有第三类特征，如用户浏览/购买品类，有多个品类id且用一个数值衡量用户浏览或购买每个品类商品的数量。这类特征按照categorical特征处理，不同的只是特征的值不是0或1，而是代表用户浏览或购买数量的数值。按前述方法得到field_id之后，再对转换后特征顺序编号，得到feat_id，特征的值也可以按照之前的方法获得。

CTR、CVR预估样本的类别是按不同方式获取的。CTR预估的正样本是站内点击的用户-商品记录，负样本是展现但未点击的记录；CVR预估的正样本是站内支付（发生转化）的用户-商品记录，负样本是点击但未支付的记录。构建出样本数据后，采用FFM训练预估模型，并测试模型的性能。

|         | #(field) | #(feature) | AUC  | Logloss |
| :------ | :------- | :--------- | :--- | :------ |
| 站内CTR | 39       | 2456       | 0.77 | 0.38    |
| 站内CVR | 67       | 2441       | 0.92 | 0.13    |

由于模型是按天训练的，每天的性能指标可能会有些波动，但变化幅度不是很大。这个表的结果说明，站内CTR/CVR预估模型是非常有效的。

在训练FFM的过程中，有许多小细节值得特别关注。

第一，样本归一化。FFM默认是进行样本数据的归一化，即 \( pa.norm \) 为真；若此参数设置为假，很容易造成数据inf溢出，进而引起梯度计算的nan错误。因此，样本层面的数据是推荐进行归一化的。

第二，特征归一化。CTR/CVR模型采用了多种类型的源特征，包括数值型和categorical类型等。但是，categorical类编码后的特征取值只有0或1，较大的数值型特征会造成样本归一化后categorical类生成特征的值非常小，没有区分性。例如，一条用户-商品记录，用户为“男”性，商品的销量是5000个（假设其它特征的值为零），那么归一化后特征“sex=male”（性别为男）的值略小于0.0002，而“volume”（销量）的值近似为1。特征“sex=male”在这个样本中的作用几乎可以忽略不计，这是相当不合理的。因此，将源数值型特征的值归一化到 \( [0, 1] \) 是非常必要的。

第三，省略零值特征。从FFM模型的表达式可以看出，零值特征对模型完全没有贡献。包含零值特征的一次项和组合项均为零，对于训练模型参数或者目标值预估是没有作用的。因此，可以省去零值特征，提高FFM模型训练和预测的速度，这也是稀疏样本采用FFM的显著优势。

本文主要介绍了FFM的思路来源和理论原理，并结合源码说明FFM的实际应用和一些小细节。从理论上分析，FFM的参数因子化方式具有一些显著的优势，特别适合处理样本稀疏性问题，且确保了较好的性能；从应用结果来看，站内CTR/CVR预估采用FFM是非常合理的，各项指标都说明了FFM在点击率预估方面的卓越表现。