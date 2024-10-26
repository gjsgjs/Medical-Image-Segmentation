from sklearn.mixture import GaussianMixture
from skimage.segmentation import active_contour
from skimage.filters import gaussian
import cv2
import numpy as np
import torch.nn as nn

# ground_truth阈值分割
def fixed_threshold_segmentation(image, threshold=127):
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

# 大津阈值分割
def otsu_threshold_segmentation(image):
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_image = cv2.bitwise_not(binary_image)
    return binary_image

# 形态学处理
def morphological_processing(image, kernel_size=5,iterations=1,set = 'close'):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if set == 'close':
        # 膨胀
        dilated_image = cv2.dilate(image, kernel, iterations=iterations)
        # 腐蚀
        eroded_image = cv2.erode(dilated_image, kernel, iterations=iterations)
        out_image = eroded_image
    if set == 'open':
        # 腐蚀
        eroded_image = cv2.erode(image, kernel, iterations=iterations)
        # 膨胀
        dilated_image = cv2.dilate(eroded_image, kernel, iterations=iterations)
        out_image = dilated_image

    return out_image

# k-means聚类分割
def kmeans_segmentation(image, k=2):
    # 将图像转换为一维数组
    pixel_values = image.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)

    # 定义K-means聚类的标准
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # 应用K-means聚类
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

 
   
    # 将中心值转换为0和255
    centers = np.uint8(centers)
    if centers[0] > centers[1]:
        centers[0] = 255
        centers[1] = 0
    else:
        centers[0] = 0
        centers[1] = 255
    # 将每个像素转换为其中心值
    segmented_image = centers[labels.flatten()]

    # 将图像转换回原始形状
    segmented_image = segmented_image.reshape(image.shape)
    # 反转图像
    segmented_image = cv2.bitwise_not(segmented_image)
    return segmented_image


def gmm_segmentation(image, n_components=2):
    # 将图像转换为二维数组
    pixel_values = image.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)

    # 定义GMM模型
    gmm = GaussianMixture(n_components=n_components, random_state=0)

    # 拟合GMM模型
    gmm.fit(pixel_values)

    # 预测每个像素的类别
    labels = gmm.predict(pixel_values)

    # 将每个像素转换为其类别中心值
    centers = np.uint8(gmm.means_)
    # 将中心值转换为0和255
    centers = np.uint8(centers)
    if centers[0] > centers[1]:
        centers[0] = 255
        centers[1] = 0
    else:
        centers[0] = 0
        centers[1] = 255
    
    segmented_image = centers[labels]

    # 将图像转换回原始形状
    segmented_image = segmented_image.reshape(image.shape)
    # 反转图像
    segmented_image = cv2.bitwise_not(segmented_image)

    return segmented_image


def graph_cut_segmentation(image):
    # # 将灰度图像转换为彩色图像
    # color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) # (256,256,3)

    # 初始化掩码
    mask = np.zeros(image.shape[:2], np.uint8) # (256,256) 0确定的背景 1确定的前景 2可能的背景 3可能的前景


    # 创建背景和前景模型
    bgd_model = np.zeros((1, 65), np.float64) 
    fgd_model = np.zeros((1, 65), np.float64)

    # 定义矩形区域(初始的前景区域) 初始化前景大小很重要
    rect = (10, 10, image.shape[1] - 20, image.shape[0] - 20)
    # rect = (50, 50, image.shape[1] - 100, image.shape[0] - 100)

    # 应用GrabCut算法
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # 将掩码修改为二值掩码
    mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8') # 确定的背景和可能的背景都设置为背景
    # mask2 = np.where((mask == 1) , 255, 0).astype('uint8')
    # segmented_image = image * mask2

    return mask2


def active_contour_segmentation(image):
    # 平滑图像
    smoothed_image = gaussian(image, 2)

    # 初始化轮廓 初始参数大的100,小的50
    s = np.linspace(0, 2 * np.pi, 800)
    x = 128 + 100 * np.cos(s)
    y = 128 + 100 * np.sin(s)
    init = np.array([x, y]).T

    # 应用主动轮廓模型
    snake = active_contour(smoothed_image, init, alpha=0.015, beta=10, gamma=0.001)



    # 创建一个空白图像用于绘制轮廓
    segmented_image = np.zeros_like(image)
    # 将轮廓转换为整数坐标
    snake = np.round(snake).astype(int)
    # 交换 snake 数组中每个元素的第一个和第二个值
    snake[:, [0, 1]] = snake[:, [1, 0]]
    # 填充轮廓内部
    cv2.fillPoly(segmented_image, [snake], 255)

    # # 创建一个空白图像用于绘制轮廓
    # segmented_image = np.zeros_like(image)
    # for i in range(len(snake)):
    #     segmented_image[int(snake[i, 0]), int(snake[i, 1])] = 255

    return segmented_image



# 定义UNet模型
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 输入: (3, 256, 256), 输出: (64, 256, 256)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 输出: (64, 256, 256)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)         # 输出: (64, 128, 128)
        )
        
        # 中间层
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 输出: (128, 128, 128)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 输出: (128, 128, 128)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)           # 输出: (128, 64, 64)
        )
        
        # 解码器部分
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 输出: (64, 128, 128)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),           # 输出: (64, 128, 128)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),   # 输出: (64, 256, 256)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),           # 输出: (64, 256, 256)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)                        # 输出: (1, 256, 256)
        )
        
    def forward(self, x):
        # 前向传播 x.shape = [N, 3, 256, 256]
        x1 = self.encoder(x)  # x1.shape = [N, 64, 128, 128]
        x2 = self.middle(x1)  # x2.shape = [N, 128, 64, 64]
        x3 = self.decoder(x2)  # x3.shape = [N, 1, 256, 256]
        return x3