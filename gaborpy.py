import cv2
import numpy as np
from skimage import filters
# skimage.filters.gabor（）函数返回的是图像变换后的实部和虚部，在图像识别领域一般使用其模作为图像特征
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
