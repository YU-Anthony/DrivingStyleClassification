---
title: Driving style classification and predication
mathjax: true
---

### Introduction
传统的分类方法依赖于特征工程和机器学习算法，比如主成分分析法（ Principal component analysis， PCA）[<sup>1</sup>](#refer-anchor-1)以无监督的方式检测出5个不同的驾驶类别。[<sup>2</sup>](#refer-anchor-2)使用了基于GMM的驾驶员模型来识别单个驾驶员的行为，主要研究了跟车行为和踏板操作（pedal operation）。[<sup>3</sup>](#refer-anchor-3)使用词袋（Bag-of-words）和K均值聚类来表示个体的驾驶特征。[<sup>4</sup>](#refer-anchor-4)使用了一个自编码网(autoencoder network)来提取基于道路类型的驾驶特征。

然而，这些传统方法主要依赖 `handcrafted features`，这限制了机器学习算法获得更好的性能; 为此，我们可以考虑使用深度学习的方法。与现有方法相比，深度学习方法可以大大减少人工操作。

2014 年, Zheng J [<sup>5</sup>](#refer-anchor-5) 等提出了一个普通的前馈神经网络模型来预测驾驶员的换道行为, 该模型融合了多个车辆在行驶过程中的数据特征, 其准确率达到了94.58%(左换道) 和73.33%(右换道)。 但是该模型没有考虑驾驶员自身的生理特征并且右换和左换道的准确率不对称. 2015 年, Peng JS 等[<sup>6</sup>](#refer-anchor-6)提出了一种多特征融合的神经网络的模型, 该模型考虑了驾驶员例如头部转动等生理因素。 并利用前馈神经网络来预测驾驶员的换道意图。但文章中提到该模型的时间窗口提取方法不准确, 可能会影响模型性能。2016 年, Dou YL 等[<sup>7</sup>](#refer-anchor-7)提出了一种以支持向量机和前馈神经网络为主要方法的预测模型. 该模型只对支持向量机和前馈神经网络的预测结果做了分析和合并, 并没有将支持向量机和神经网络的结构加以改变。 同年, Wang XP 等[<sup>8</sup>](#refer-anchor-8)提出了一种以前馈神经网络和卷积神经网络为基础的混合神经网络模型, 该模型以时间序列为输入, 通过卷积网络来提取数据特征, 再经过前馈网络进行数据融合后得出结果。但此模型的问题在于当用卷积网络提取数据时, 可能会造成数据的损失, 使得预测精度降低。

### Data Preprocess


### Methods
#### 1. CNN
卷积神经网络（Convolutional Neural Network，CNN）是一种适合使用在连续值输入信号上的深度神经网络，比如声音、图像和视频。

##### 建议阅读文章
1. Characterizing Driving Styles with Deep Learning
[Paper](https://arxiv.org/pdf/1607.03611.pdf) / [Code](https://github.com/sobhan-moosavi/CharacterizingDrivingStylesWithDeepLearning)

#### 2. Seq2Seq
Seq2Seq, 是普通 RNN 的增强版本, 由一个编码器 Encoder 和一个解码器Decoder 组成, 而 Encoder 和 Decoder 的计算内核由 GRU 构成。它能够很好的处理时间序列数据，在图像、语音和 NLP，比如：机器翻译、机器阅读、语音识别、智能对话和文档摘要生成等，都有广泛的应用。这也是目前一个不之处，seq2seq 在分类方面用的较少

大多数场景下使用的 Seq2Seq 模型基于 RNN 构成的，虽然取得了不错的效果，但也有一些学者发现使用 CNN 来替换 Seq2Seq 中的 encoder 或 decoder可以达到更好的效果。但实际上 CNN-seq2seq 模型 不一定比 LSTM-Seq2Seq 好。采用 CNN 的 Seq2Seq 最大的优点在于速度快，效率高，缺点就是需要调整的参数太多。上升到 CNN 和 RNN 用于 NLP 问题时，CNN 也是可行的，且网络结构搭建更加灵活，效率高，特别是在大数据集上，往往能取得比RNN更好的结果。

##### 建议阅读文章
######1. Time series forecasting with deep stacked unidirectional and bidirectional LSTMs
[Blog](https://towardsdatascience.com/time-series-forecasting-with-deep-stacked-unidirectional-and-bidirectional-lstms-de7c099bd918) **/** [Code](https://github.com/manohar029/TimeSeries-Seq2Seq-deepLSTMs-Keras/blob/master/Keras_Enc-Dec_MinTempMel.ipynb)

######2. Python机器学习笔记：利用 Keras 进行分类预测
[Blog](https://www.cnblogs.com/wj-1314/p/9591369.html)

######3. seq2seq 的 keras 实现
[Blog](https://cloud.tencent.com/developer/article/1083503)
### Problems

驾驶风格目前还没有一个准确的定义，因此分类的依据也有很多种，比如油耗，均速，跟车行为等。一般来说，对驾驶风格的分类大多是将其分为若干类，对应于不同的离散值，或许可以考虑连续型的驾驶风格分类算法，比如将其描述为介于-1到+1之间的值。

### Reference

<div id="refer-anchor-1"></div>

[1] [Z. Constantinescu, C. Marinoiu, and M. Vladoiu, “Driving style analysis using data mining techniques,” International Journal of Computers Communications & Control, vol. 5, no. 5, pp. 654–663, 2010](https://ieeexplore.ieee.org/abstract/document/8947318)

<div id="refer-anchor-2"></div>

[2] [C. Miyajima, Y. Nishiwaki, K. Ozawa, T. Wakita, K. Itou, K. Takeda, and F. Itakura, “Driver modeling based on driving behavior and its evaluation in driver identification,” Proceedings of the IEEE, vol. 95, no. 2, pp. 427–437, 2007](https://ieeexplore.ieee.org/document/8968478/)

<div id="refer-anchor-3"></div>

[3] [E. Yurtsever, C. Miyajima, S. Selpi, and K. Takeda, “Driving signature extraction,” in FAST-zero’15: 3rd International Symposium on Future Active Safety Technology Toward zero traffic accidents, 2015, 2015.](https://ieeexplore.ieee.org/document/8317860)

<div id="refer-anchor-4"></div>

[4] [K. Sama, Y. Morales, N. Akai, H. Liu, E. Takeuchi, and K. Takeda Driving feature extraction and behavior classification using an autoencoder to reproduce the velocity styles of experts,” in 2018 21st International Conference on Intelligent Transportation Systems (ITSC). IEEE, 2018, pp. 1337–1343.](https://ieeexplore.ieee.org/document/8317860)

[5] [Zheng J, Suzuki K, Fujita M. Predicting driver’s lanechanging decisions using a neural network model. Simulation Modelling Practice and Theory, 2014, 42: 73–83.](https://ieeexplore.ieee.org/document/8317860)

[6] [Peng JS, Guo YS, Fu R, et al. Multi-parameter prediction of drivers’ lane-changing behaviour with neural network model.Applied Ergonomics, 2015, 50: 207–217.](https://ieeexplore.ieee.org/document/8317860)

[7] [Dou YL, Yan FJ, Feng DW. Lane changing prediction at highway lane drops using support vector machine and artificial neural network classifiers. Proceedings of 2016 IEEE International Conference on Advanced Intelligent Mechatronics. Banff, PB, Canada. 2016. 901–906.](https://ieeexplore.ieee.org/document/8317860)

[8] [Wang XP, Murphey YL, Kochhar DS. MTS-DeepNet for lane change prediction. Proceedings of 2016 International Joint Conference on Neural Networks. Vancouver, BC, Canada. 2016. 4571–4578.](https://ieeexplore.ieee.org/document/8317860)