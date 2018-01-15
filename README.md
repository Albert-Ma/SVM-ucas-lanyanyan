# SVM-ucas-lanyanyan
2018年国科大模式识别与机器学习兰艳艳-SVM  
@2018/15  
@authon xy

# 1.History
创始人及主要的贡献人Vapnik  
Linear SVM-> Kernelized SVM-> SVR...SMO...  
# 2.从Linear线性分类器说起
对于二维平面上二类划分问题  
![](https://github.com/Albert-xy/SVM-ucas-lanyanyan/blob/master/imp/linear-clasifier.png)  
选择分类面的标准：**泛化误差最小**，可以简单理解为要使得两类之间间隔足够大  
黑色那条就是最优的分类线，如何以数学的形式刻画这个最优的分类超平面？间隔怎么定义？  
#  3.Margin(边际或者边距)的来源
对于logistic 回归问题$$P(y=1|x)=$$
<img src="http://www.forkosh.com/mathtex.cgi? \Large P(y=1|x)=\cfrac{1}{1+e^{-w^t*x}}">
