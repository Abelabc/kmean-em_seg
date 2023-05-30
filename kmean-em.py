import io

from sklearn.mixture import GaussianMixture
from math import floor
from sklearn.preprocessing import MinMaxScaler
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans

def image_patch(img, slide_window, h, w):
	# 滑动窗口
    patch = np.zeros((slide_window, slide_window, h, w), dtype=np.uint8)

    for i in range(patch.shape[2]):
        for j in range(patch.shape[3]):
            patch[:, :, i, j] = img[i: i + slide_window, j: j + slide_window,1]

    return patch

def calcu_glcm(img, slide_window ):
    vmin = 0
    vmax = 255
    nbit = 64
    step = [2]
    angle = [0]
    # 计算GLCM矩阵
    mi, ma = vmin, vmax
    h, w = img.shape[:2]

    bins = np.linspace(mi, ma + 1, nbit + 1)
    img = np.digitize(img, bins) - 1

    img = cv2.copyMakeBorder(img, floor(slide_window / 2), floor(slide_window / 2)
                              , floor(slide_window / 2), floor(slide_window / 2), cv2.BORDER_REPLICATE)  # 图像扩充

    patch = image_patch(img, slide_window, h, w)

    glcm = np.zeros((nbit, nbit, len(step), len(angle), h, w), dtype=np.uint8)
    for i in range(patch.shape[2]):
        for j in range(patch.shape[3]):
            glcm[:, :, :, :, i, j] = graycomatrix(patch[:, :, i, j], step, angle, levels=nbit)

    return glcm

def calcu_glcm_Auto_correlation(glcm, nbit=64):
    Auto_correlation = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            Auto_correlation += glcm[i, j] * i * j

    return Auto_correlation


def calcu_glcm_mean(glcm, nbit=64):
    mean = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i, j] * i / (nbit) ** 2

    return mean


def calcu_glcm_con(glcm, nbit=64):
    con = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            con += ((i - j) ** 2) * glcm[i, j]
    return con


def calcu_glcm_entropy(glcm, nbit=64):
    eps = 0.00001
    entropy = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            entropy -= glcm[i, j] * np.log10(glcm[i, j] + eps)

    return entropy


def calcu_glcm_asm(glcm, nbit=64):
    asm = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            asm += glcm[i, i] ** 2

    return asm


def calcu_glcm_h(glcm, nbit=64):
    h = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            h += glcm[i, j] / (1 + (i - j) ** 2)
    return h


def calcu_glcm_correlation(glcm, nbit=64):
    mean = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i, j] * i / (nbit) ** 2

    variance = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            variance += glcm[i, j] * (i - mean) ** 2

    correlation = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            correlation += ((i - mean) * (j - mean) * (glcm[i, j] ** 2)) / variance

    return correlation


if __name__ == '__main__':
    #img = np.array(Image.open('mosaic A.bmp'))
    img = cv2.imread('mosaic A.bmp')
    img = np.uint8(255.0 * (img - np.min(img)) / (np.max(img) - np.min(img)))
    h, w = img.shape[:2]
    # height, width = img.shape[:2]

    glcm = calcu_glcm(img,9)
    for i in range(glcm.shape[2]):
        for j in range(glcm.shape[3]):
            glcm_cut = glcm[:, :, i, j, :, :]

            # 获取性能评价指标 shape(168,168)
            mean = calcu_glcm_mean(glcm_cut)
            # con = calcu_glcm_con(glcm_cut)
            con = graycoprops(glcm_cut, 'contrast')
            # asm = calcu_glcm_asm(glcm_cut)
            asm = graycoprops(glcm_cut, 'ASM')
            # ent = calcu_glcm_entropy(glcm_cut)
            ent = graycoprops(glcm_cut, 'energy')
            # h = calcu_glcm_h(glcm_cut)
            h = graycoprops(glcm_cut, 'homogeneity')
            # corr = calcu_glcm_correlation(glcm_cut)
            corr = graycoprops(glcm_cut, 'correlation')
            dis = graycoprops(glcm_cut, 'dissimilarity')

    data = np.array([mean, con, asm, ent, h, dis]).transpose((1, 2, 0))
    data = data.reshape((-1, 6))
    min_max_sc = MinMaxScaler()
    data = min_max_sc.fit_transform(data)
    kmeans = KMeans(n_clusters=4, n_init= 'auto', max_iter=100)
    kmeans.fit(data)
    centers = kmeans.cluster_centers_
    labels = kmeans.predict(data)
    labels = labels.reshape((256, 256))
    means_init = centers
    plt.imshow(labels, cmap='gray')
    plt.savefig('kmean.png', bbox_inches='tight', pad_inches=0)
    plt.show()

    # 构建GMM模型
    gmm = GaussianMixture(n_components=4, means_init=means_init)
    # 训练模型
    gmm.fit(data)

    # 预测每个像素点属于哪个簇
    labels = gmm.predict(data)
    labels = labels.reshape((256, 256))

    plt.imshow(labels, cmap='gray')
    plt.axis('off')
    plt.savefig('em.png', bbox_inches='tight', pad_inches=0)
    plt.show()
