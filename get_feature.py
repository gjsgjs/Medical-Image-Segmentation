import cv2
import numpy as np

def pix_preprocess_image(image):
    # 高斯平滑
    smoothed_image = cv2.GaussianBlur(image, (3, 3), 0)
    
    # # 直方图均衡化
    # equalized_image = cv2.equalizeHist(smoothed_image)
    
    # 自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    equalized_image = clahe.apply(smoothed_image)

    return equalized_image

def gabor_filter(image, ksize=21, sigma=4, theta=0, lambd=10.0, gamma=0.5, psi=0):
    # 创建 Gabor 滤波器
    gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
    
    # 应用 Gabor 滤波器
    filtered_image = cv2.filter2D(image, cv2.CV_32F, gabor_kernel)
    filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)
    filtered_image = np.uint8(filtered_image)
    return filtered_image

def Texture_preprocess_image(image):
    # 提取纹理特征
    gabor_features = []
    for theta in np.arange(0, np.pi, np.pi / 8):
        filtered_image = gabor_filter(image, theta=theta)
        gabor_features.append(filtered_image)
    
    # 所有 Gabor 滤波器响应的同一位置上取最大值来生成最终的纹理图像
    # texture_image = np.max(gabor_features, axis=0)
    # 所有 Gabor 滤波器响应的同一位置上取平均值来生成最终的纹理图像
    texture_image = np.mean(gabor_features, axis=0)

    texture_image = np.uint8(texture_image)
    return texture_image


def laws_kernels():
    # 定义Laws' Kernel
    L5 = np.array([1, 4, 6, 4, 1])
    E5 = np.array([-1, -2, 0, 2, 1])
    S5 = np.array([-1, 0, 2, 0, -1])
    R5 = np.array([1, -4, 6, -4, 1])
    W5 = np.array([-1, 2, 0, -2, 1])
    # l = [1,2,1]
    # e = [-1,0,1]
    # s = [-1,2,-1]

    kernels = []
    for kernel1 in [L5,E5, S5, R5,W5]:
        for kernel2 in [L5,E5, S5, R5, W5]:
            kernels.append(np.outer(kernel1, kernel2))
    return kernels

def apply_laws_kernels(image, kernels):
    # 应用Laws' Kernel
    texture_features = []
    # import pdb; pdb.set_trace()
    for kernel in kernels:
        filtered_image = cv2.filter2D(image, cv2.CV_32F, kernel)
        # 取绝对值
        filtered_image = np.abs(filtered_image)

        # 正则化
        # filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)
        # filtered_image = np.uint8(filtered_image)

        # cv2.imshow('filtered_image', filtered_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        texture_features.append(filtered_image)
    return texture_features

def Laws_preprocess_image(image):
   
    # 获取Laws' Kernel
    kernels = laws_kernels()
    
    # 应用Laws' Kernel提取纹理特征
    texture_features = apply_laws_kernels(image, kernels)
    
    # 将所有纹理特征叠加在一起
    # texture_image = np.max(texture_features, axis=0)
    # import pdb; pdb.set_trace()
    # texture_image = np.mean(texture_features, axis=0)


    texture_image = np.sum(texture_features, axis=0)
    texture_image = cv2.normalize(texture_image, None, 0, 255, cv2.NORM_MINMAX)
    texture_image = np.uint8(texture_image)
    
    return texture_image