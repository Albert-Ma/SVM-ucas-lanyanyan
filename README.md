# SVM-ucas-lanyanyan
2018年国科大模式识别与机器学习兰艳艳-SVM  
@2018/15  
@authon xy

# 1.History
创始人及主要的贡献人Vapnik  
Linear SVM-> Kernelized SVM-> SVR...SMO...  
# 2.从Linear线性分类器说起
对于二维平面上二类划分问题  
![图1](https://github.com/Albert-xy/SVM-ucas-lanyanyan/blob/master/imp/linear-clasifier.png)  
- 选择分类面的标准：**泛化误差最小**，可以简单理解为要使得两类之间间隔足够大   
- 黑色那条就是最优的分类线，Q1间隔怎么定义？Q2如何以数学的形式刻画这个最优的分类超平面？  
#  3.Margin(边际或者边距)的来源  
以logistic回归问题为例  
![图2](https://github.com/Albert-xy/SVM-ucas-lanyanyan/blob/master/imp/LR-1.png)    
- P(y=1|x)越大，我们越有**自信**说x对应y的lable为1，这里是基于概率上的**自信**  
- 在一个没有概率的分类器上应该怎么定义**Q1**，从图1上来看**自信**就对应点到分类面的距离   
- **自信**其实的就是分类面的边距即margin
  
## 函数间距  
- 回到2.线性分类器上：f(x)=wx+b,f(x)>0,lable=1(正例);f(x)<0,lable=-1(反例)   
- 分类的超平面(实际是条线):wx+b=0
- ___γ=min y(wx+b)___,点到分类面的距离**最小**的那个，这些点也称为支持向量，后面说  
![](https://github.com/Albert-xy/SVM-ucas-lanyanyan/blob/master/imp/LR-2.png)  
## 几何间距  
### 简单的推导（徐君）：  

![](https://github.com/Albert-xy/SVM-ucas-lanyanyan/blob/master/imp/LR-3.png)      
- ___wx1+b = 1,wx2+b = -1___ => ___w(x1-x2) = 2___,以x1(向量)替代x+，x2替代x-___<1>___
- ___x1 = x2+λw => x1-x2 = λw___ ___<2>___
- 由1和2推出：2/w ___=λw => λ = 2/(w)^2___
- 分类间隔Margin = ___|x1-x2| = |λw| = 2/(w)^2 * |w|= 2/|w|___  
- ___最大化M即max 2/|w| 等价于 min|w|^2___
### 简单的推导（兰艳艳）：  
![](https://github.com/Albert-xy/SVM-ucas-lanyanyan/blob/master/imp/LR-5.png) 
- 点A（xi,yi）,A到分类面的边距γi（向量）
- 点B用A表示xj=___xi-γi(w/|w|2)，将其代入wx+b=0得
- ___w(xi-γi(w/|w|2))+b=0 => γi=(w/|w|2)xi+b/|w|2___  
- ___γi可正可负，最终的γi = yi((w/|w|2)xi+b/|w|2)___
- ___γ=min γi___
#### 函数边际和几何边际的关系
- ___γ = γ^/(|w|2)当|w|=1时两者相等___
- ___max γ <=> max γ^/(|w|2),___ s.t.
### 最后的优化margin分类器
![](https://github.com/Albert-xy/SVM-ucas-lanyanyan/blob/master/imp/LR-4.png)  
