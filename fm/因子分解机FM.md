# 因子分解机FM

## 特征组合

对逻辑回归最朴素的特征组合就是二阶笛卡尔乘积，但是这样有几个问题：

1. 两两组合导致特征维度灾难；
2. 组合后的特征不见得都有效，事实上大部分可能无效；
3. 组合后的特征样本非常稀疏，即并不能在样本中找到对应的组合，也就没办法更新参数。

如果把包含了特征两两组合的逻辑回归线性部分写出来，就是：
$$
\hat{y}=\omega_{0}+\sum_{i=1}^{n} \omega_{i} x_{i}+\sum_{i=1}^{n} \sum_{j=i+1}^{n} \omega_{i j} x_{i} x_{j}
$$
针对这个问题，就有了一个新的模型：**因子分解机模型**，也叫作FM（Factorization Machine）。

### FM模型原理

因子分解机模型在2010年被提出来，因为逻辑回归在做特征组合时样本稀疏，从而无法学到很多特征组合的权重，所以能不能把上面公式的$w_{ij}$做解耦，让**每一个特征学习一个隐因子向量出来。注意这里的特征指的是onehot之后的特征。也就是说FM把一个类别特征转换成了向量形式，只不过这个向量会根据特征的取值而变化。**

公式如下：
$$
\hat{y}=\omega_{0}+\sum_{i=1}^{n} \omega_{i} x_{i}+\sum_{i=1}^{n} \sum_{j=i+1}^{n}<V_{i}, V_{j}>x_{i} x_{j}
$$
化简后：
$$
\hat{y}=\omega_{0}+\sum_{i=1}^{n} \omega_{i} x_{i}+\frac{1}{2} \sum_{f=1}^{k}\left(\left(\sum_{i=1}^{n} v_{i, f} x_{i}\right)^{2}-\sum_{i=1}^{n} v_{i, f}^{2} x_{i}^{2}\right)
$$


向量形式：
$$
\hat y_i = w_0+w^Tx_i+\frac12\sum_{f=1}^k((v_f^Tx_i)^2-(v_f^T)^2x_i^2)
$$


$x_i$是n维列向量，表示一个样本，$v_f$是n维列向量，表示隐因子第$f$个维度对应的特征

矩阵形式就是：
$$
\hat Y = w_0+Xw+\frac12*sum((XV)^2-X^2V^2,axis=1)
$$
设样本数为$m$，特征维度为$n$，隐因子维度是$k$，则$X\in R_{m \times n},V\in R_{n\times k}$ ，**这里样本矩阵$X$每一行表示一个样本。**

$\hat y$对$v_f$的梯度：
$$
\begin{align}
&\frac{\partial \hat y_i}{\partial v_{jf}} = (\sum_{j=1}^n v_{jf}x_{ij})x_{ij}- x_{ij}^2v_{jf} \\
&\frac{\partial \hat y_i}{\partial v_f} = x_i^Tv_fx_i-x_i^2\odot v_f \\
&\frac{\partial \hat y_i}{\partial V} = x_ix_i^TV - x_i^2\odot V
\end{align}
$$


如果是平方损失函数：
$$
L=\frac{1}{2}\Sigma_{i=1}^m(y_i-\hat y_i)^2
$$
梯度为：
$$
\begin{align}
&\frac{\partial L}{\partial w}=-\Sigma_{i=1}^m(y_i-\hat y_i)x_i \\
&\frac{\partial L}{\partial v_f}=-\Sigma_{i=1}^m(y_i-\hat y_i)\frac{\partial \hat y_i}{\partial v_f}
\end{align}
$$

二分类问题将$\hat y$输入到$sigmoid$函数作为最终的概率，然后用交叉熵损失函数即可：
$$
\begin{align}
&\frac{\partial L}{\partial w}=-\Sigma_{i=1}^m(y_i-\sigma (\hat y_i))x_i \\
&\frac{\partial L}{\partial v_f}=-\Sigma_{i=1}^m(y_i-\sigma (\hat y_i))\frac{\partial \hat y_i}{\partial v_f}
\end{align}
$$
