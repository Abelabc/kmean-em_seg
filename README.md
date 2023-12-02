# kmeans算法和em算法的结合进行图像分割
![](https://img.shields.io/github/stars/pandao/editor.md.svg) ![](https://img.shields.io/github/forks/pandao/editor.md.svg) ![](https://img.shields.io/github/tag/pandao/editor.md.svg) ![](https://img.shields.io/github/release/pandao/editor.md.svg) ![](https://img.shields.io/github/issues/pandao/editor.md.svg) ![](https://img.shields.io/bower/v/editor.md.svg)
## 任务要求
给出了两个马赛克图像（mosaicA.bmp 和 mosaicB.bmp），利用kmeans和em图像分割方法进行分割，还提供了地图（mapA.bmp 和 mapB.bmp）用于数值比较。这涉及的特征可以是 Gabor 提取的多维特征向量过滤器组。您可以微调方向数（例如，8）和刻度（例如，5）得到最佳分割性能（最高分类精度）。

[![](https://github.com/Abelabc/kmean-em_seg/blob/main/mosaic%20A.bmp)](https://github.com/Abelabc/kmean-em_seg/blob/main/mosaic%20A.bmp "mosaic A.bmp")

> 图为：mosaic A.bmp

[![](https://github.com/Abelabc/kmean-em_seg/blob/main/mapA.bmp)](https://github.com/Abelabc/kmean-em_seg/blob/main/mapA.bmp "mapA.bmp")

> 图为：mapA.bmp

## 图像生成
### 1.导入库
```
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
```
### 2.用K-means算法进行聚类查看效果，k=4，误差控制0.0001，得到kmean.png

![](https://github.com/Abelabc/kmean-em_seg/blob/main/kmean.png)

### 3.使用 kmean 算法的聚类中心初始化 em 算法来训练高斯混合模型
```
centers = kmeans.cluster_centers_
means_init = centers
# 构建GMM模型
gmm = GaussianMixture(n_components=4, means_init=means_init)
# 训练模型
gmm.fit(data)
```
em.png：

![](https://github.com/Abelabc/kmean-em_seg/blob/main/em.png)

## 图像数值比较
### 1.通过定义 gaborcls 函数计算原始 mapB.bmp 的特征值
```
def gaborcls(filename):
    img = cv2.imread(filename)  # 读图像
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转灰度
    frequency = 0.6
    # gabor变换
    real, imag = filters.gabor(img_gray, frequency=0.6, theta=60, n_stds=5)
    # 取模
    img_mod = np.sqrt(real.astype(float) ** 2 + imag.astype(float) ** 2)
    # 图像缩放（下采样）
    newimg = cv2.resize(img_mod, (0, 0), fx=1 / 4, fy=1 / 4, interpolation=cv2.INTER_AREA)
    tempfea = newimg.flatten()  # 矩阵展平
    tmean = np.mean(tempfea)  # 求均值
    tstd = np.std(tempfea)  # 求方差
    newfea = (tempfea - tmean) / tstd  # 数值归一化
    return newfea
```
这段代码会得到一列特征量，我们对前后特征量进行余弦操作cosine

### 余弦结果比较
```
//对组合分割效果进行分析
filename = 'em.png'
newfea = gaborcls(filename)
tmp=cosine(newfea,fea)
print(newfea)
print(tmp)
```
```
//对kmeans单独分割效果进行分析
filename = 'kmean.png'
newfea = gaborcls(filename)
print(newfea.shape)
tmp=cosine(newfea,fea)
print(newfea)
print(tmp)
```
## 结果分析

通过计算余弦相似度比较和mapA.bmp效应的差异，通过比较，一般来说，结合kmean和em算法确实有更好的效果，得到的结果明显优于单独的kmean算法。
