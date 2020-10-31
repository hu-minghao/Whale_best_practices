# AI  best practices
本项目旨在对AI学习提供最佳实践项目😀，目前提供一个最基础的图像分类教程，旨在从数据准备，到部署上线
的整套流程
🏍

教程在陆续收集中，感兴趣的可以贡献自己的项目案例我😆😆😆

# 1.图像处理 👤

图像处理任务可分为三个方向,即图像分类,目标检测和图像分割

- 分类任务

把多种图像分类识别出来

最优的代表网络:resnet网络和VGG系列网络

例如把猫狗进行二分类

- 目标检测

把目标图像用回归框框出进行类别识别,对目标检测的网络可以分为两大派:一次检测和两次检测

两次检测的最优的代表网络是faster rcnn系列网络,一次检测的代表网络是yolo系列网络

一次检测在保证识别准确率的情况下,检测速度相对较快,因此yolo系列的网络用的较多

- 图像分割

把目标图像的轮廓分割出来,分为语义分割,实例分割,全景分割


# 2.自然语言处理 👓 

自然语言处理主要有以下任务

- 词干抽取（Stemming）

词干抽取是去除词语的形态等，将词语变成它的词干形式的过程。它的目的是将不同形式的相同含义的词语统一起来（数据学习网站提醒一下：中文中一般没有词干抽取的工作，但是多了个分词，这是中英文文本处理的不同）。

- 词形还原（Lemmatisation）

词形还原是指将一组词语变成他们词干的形式的过程。例如在会话识别任务中，我们需要考虑这个单词在句子中的含义，也要考虑这个单词在相邻句子中的含义。

- 词嵌套（Word Embeddings）

词嵌套是一种技术，它可以将自然语言变成实值向量的形式。由于计算机无法直接处理文本，所以这种转换很有用。这类技术使用实值向量来表示自然语言中词语的位置和
相互关系（数据学习网站提醒一下：词嵌套最有名的论文应当属于word2vec这篇论文，它并没有说提供了新方法，但是提供了一种新工具，可以很方便的从文本中获取词向量的结果。
这也是谷歌提出来的，谷歌真是个不错的公司）。


- 词性标注（Part-Of-Speech Tagging）

词性标注就是将句子中的单词标注成“名词”、“动词”等（数据学习网站提醒一下：中文的词性标注工具可以使用结巴分词或者是张华平分词，都是带有词性标注的功能）。


- 命名实体消歧（Named Entity Disambiguation）

命名实体消岐是值识别句子中的实体的过程。例如，句子：
“Apple earned a revenue of 200 Billion USD in 2016”
命名实体消歧的目标是认出Apple是一个公司名字而不是水果名。
命名实体一般需要一个实体库，它可以将句子中的实体链接到实体库中。


- 命名实体识别（Named Entity Recognition）

命名实体识别是要识别出句子中的实体，并将实体划分到某个类别中，例如人、组织、日期等。例如，句子：

“Ram of Apple Inc. travelled to Sydney on 5th October 2017”

返回的结果是：

Ram
of
Apple ORG
Inc. ORG
travelled
to
Sydney GPE
on
5th DATE
October DATE
2017 DATE2017 DATE

ORG表示组织，GPE表示地点。
目前命名实体识别最大的问题是，当数据变了，即使是最好的NER技术也会表现不好。

- 情感分析

情感分析的任务涉及的主题较多，一般是利用自然语言处理技术识别如客户评论中正向或者负向的情感等，或者是通过语音分析、写作分析得到情绪判别结果。例如：

“I did not like the chocolate ice-cream” – 对冰激淋做负向的评价
“I did not hate the chocolate ice-cream” – 可能是一个中立的评价
情感分析的方法很多，开始的时候可以用LSTM模型与词向量模型一起，数一数句子中正负向情感词的个数得到。资源有：

- 文本语义相似性（Semantic Text Similarity）

计算文本语义相似性就是计算两段文本之间含义相似性的任务。


- 语言识别

就是识别出文本是什么语言写的，是文本分类的一种特殊情况。


- 文本摘要（Text Summarisation）

文本摘要是通过识别文本重要的内容将一段文本缩减，并变成对这些点的总结。文本摘要的目标是最大限度保留原始文本的含义。


##  1.1 Inception-v3 图片分类器

Inception 又叫Googlenet是Google于2014年为参加ILSVRC大赛而提出的CNN分类模型。它发表于2014年的CVPR上面。在深度学习领域Google出品几乎必为精品，Inception也不例外。

它乍看上去像是蛮复杂的，但细看其结构就会发现它其实就是用一个个Inception module给堆起来的。它的设计充满了科学理论与工程实践的结合，是一个典型的data scientist与软件工程师结合搞出来的东东。

- 步骤一：下载重训练文件

查看自己的tensorflow版本，我这里是tensorflow 1.15.3，然后到tensorflow github官网下载重训练文件

[官网网址](https://github.com/tensorflow/tensorflow) 

根据自己的tensorflow的版本在branch中选择自己的版本。我这里选的1.15分支
并按照tensorflow—examples—image_retraining—retrain.py寻找即可

如何下载单个文件：打开文件，右上角的raw右键，选择链接另存为。

把retrain.py 文件放到自己的工作目录下：我这里是D:\HMH\inception_V3\

- 步骤二：数据准备

准备自己要分类的图片训练样本。

可以去 http://www.robots.ox.ac.uk/~vgg/data/ 下载数据集，

下载之后保存到指定目录（我的文件夹路径及结构如下）：

训练数据放在train文件夹下，train文件夹下，一个分类的图片，放一个文件夹下，文件夹名称以类别命名

- 步骤三：训练

修改retrain.py文件，改为自己的路径，此py文件的几处根据自己的需要更改下

修改完，点击运行retrain.py，即可自动训练。

- 步骤四：验证

运行demo_test.py，设置验证图片路径，即可对图片进行预测。

- 步骤五：部署

使用flask，对模型进行部署，将tmp文件夹复制到src文件夹下，运行main文件即可。










