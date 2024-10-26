import argparse
import os

import pandas as pd
import cv2
import numpy as np
from utils import *
from loss import evaluate_segmentation
from get_feature import *
import json

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def load_images(image_path, annotation_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    annotation = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
    return image, annotation

def load_RGB_images(image_path, annotation_path):
    image = cv2.imread(image_path)
    annotation = cv2.imread(annotation_path,cv2.IMREAD_GRAYSCALE)
    return image, annotation


# image_path = 'Data\Image\ISIC_0000000.png'  # 替换为实际路径
# annotation_path = 'Data\Annotation\ISIC_0000000.png'  # 替换为实际路径

# image, annotation = load_images(image_path, annotation_path)
# Texture_preprocessed = Laws_preprocess_image(image)
# cv2.imshow('Texture_preprocessed', Texture_preprocessed)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def get_Laws_feature(image_path, annotation_path):
    image, _ = load_images(image_path, annotation_path)
    Texture_preprocessed = Laws_preprocess_image(image)

    # Create the directory if it doesn't exist
    save_dir = os.path.join('Data', 'Laws_feature')
    os.makedirs(save_dir, exist_ok=True)
    # Extract the filename from the image path and use it for the preprocessed image
    filename = os.path.basename(image_path)
    save_path = os.path.join(save_dir, filename)

    # Save the preprocessed image
    cv2.imwrite(save_path, Texture_preprocessed)


def get_texture_feature(image_path, annotation_path):
    image, _ = load_images(image_path, annotation_path)
    Texture_preprocessed = Texture_preprocess_image(image)

    # Create the directory if it doesn't exist
    save_dir = os.path.join('Data', 'Texture_feature')
    os.makedirs(save_dir, exist_ok=True)
    # Extract the filename from the image path and use it for the preprocessed image
    filename = os.path.basename(image_path)
    save_path = os.path.join(save_dir, filename)

    # Save the preprocessed image
    cv2.imwrite(save_path, Texture_preprocessed)

    # cv2.imshow('Texture_preprocessed', Texture_preprocessed)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



def get_pix_feature(image_path, annotation_path):
    image, annotation = load_images(image_path, annotation_path)
    annotation = fixed_threshold_segmentation(annotation, threshold=127)
    pix_preprocessed = pix_preprocess_image(image)
    
    # Create the directory if it doesn't exist
    save_dir = os.path.join('Data', 'PIX_feature')
    os.makedirs(save_dir, exist_ok=True)
    # Extract the filename from the image path and use it for the preprocessed image
    filename = os.path.basename(image_path)
    save_path = os.path.join(save_dir, filename)
    
    # Save the preprocessed image
    cv2.imwrite(save_path, pix_preprocessed)
    
  



def active_contour_process_image(image_path, annotation_path):
    image, annotation = load_images(image_path, annotation_path)
    annotation = fixed_threshold_segmentation(annotation, threshold=127)
    active_contour_segmented = active_contour_segmentation(image)

    dice, sens, spec, acc, auc_score = evaluate_segmentation(annotation, active_contour_segmented)
    if args.remark:
        print('active_contour_segmented')
        print('Dice:', dice)
        print('Sensitivity:', sens)
        print('Specificity:', spec)
        print('Accuracy:', acc)
        print('AUC:', auc_score)
        print('-----------------------------')

    if args.show:
        cv2.imshow('image', image)
        cv2.imshow('annotation', annotation)
        cv2.imshow('active_contour_segmented', active_contour_segmented)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return active_contour_segmented,[dice, sens, spec, acc, auc_score]


def graph_process_image(image_path, annotation_path):
    image, annotation = load_RGB_images(image_path, annotation_path)
    annotation = fixed_threshold_segmentation(annotation, threshold=127)
    graph_cut_segmented = graph_cut_segmentation(image)

    
    dice, sens, spec, acc, auc_score = evaluate_segmentation(annotation, graph_cut_segmented)
    if args.remark:
        print('graph_cut_segmented')
        print('Dice:', dice)
        print('Sensitivity:', sens)
        print('Specificity:', spec)
        print('Accuracy:', acc)
        print('AUC:', auc_score)
        print('-----------------------------')

    if args.show:
        cv2.imshow('image', image)
        cv2.imshow('annotation', annotation)
        cv2.imshow('graph_cut_segmented', graph_cut_segmented)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return graph_cut_segmented,[dice, sens, spec, acc, auc_score]



def Clustering_process_image(image_path, annotation_path):
    image, annotation = load_images(image_path, annotation_path)
    annotation = fixed_threshold_segmentation(annotation, threshold=127)
    kmeans_segmented = kmeans_segmentation(image, k=2)
    gmm_segmented = gmm_segmentation(image, n_components=2)

    dice, sens, spec, acc, auc_score = evaluate_segmentation(annotation, kmeans_segmented)
    dice1, sens1, spec1, acc1, auc_score1 = evaluate_segmentation(annotation, gmm_segmented)
    if args.remark:
        
        print('kmeans_segmented')
        print('Dice:', dice)
        print('Sensitivity:', sens)
        print('Specificity:', spec)
        print('Accuracy:', acc)
        print('AUC:', auc_score)
        print('-----------------------------')
        print('gmm_segmented')
        print('Dice:', dice1)
        print('Sensitivity:', sens1)
        print('Specificity:', spec1)
        print('Accuracy:', acc1)
        print('AUC:', auc_score1)
        print('-----------------------------')

    if args.show:
        cv2.imshow('image', image)
        cv2.imshow('annotation', annotation)
        cv2.imshow('kmeans_segmented', kmeans_segmented)
        cv2.imshow('gmm_segmented', gmm_segmented)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return kmeans_segmented,[dice, sens, spec, acc, auc_score],gmm_segmented, [dice1, sens1, spec1, acc1, auc_score1]



def basic_process_image(image_path, annotation_path):
    image, annotation = load_images(image_path, annotation_path)
    #
    annotation = fixed_threshold_segmentation(annotation, threshold=127)
    #
    otsu_segmented = otsu_threshold_segmentation(image)
    #
    morpho_otsu_segmented = morphological_processing(otsu_segmented, kernel_size=10, iterations=2,set='close')

    dice, sens, spec, acc, auc_score = evaluate_segmentation(annotation, otsu_segmented)
    dice1, sens1, spec1, acc1, auc_score1 = evaluate_segmentation(annotation, morpho_otsu_segmented)
    if args.remark:
        
        print('otsu_segmented')
        print('Dice:', dice)
        print('Sensitivity:', sens)
        print('Specificity:', spec)
        print('Accuracy:', acc)
        print('AUC:', auc_score)
        print('-----------------------------')
        print('otsu_segmented+morphological_processing')
        print('Dice:', dice1)
        print('Sensitivity:', sens1)
        print('Specificity:', spec1)
        print('Accuracy:', acc1)
        print('AUC:', auc_score1)
        print('-----------------------------')

    if args.show:
        cv2.imshow('image', image)
        cv2.imshow('annotation', annotation)
        cv2.imshow('otsu_segmented', otsu_segmented)
        cv2.imshow('morpho_otsu_segmented', morpho_otsu_segmented)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return otsu_segmented,[dice, sens, spec, acc, auc_score],morpho_otsu_segmented, [dice1, sens1, spec1, acc1, auc_score1]

def one_way():
    config_file = 'config.json'  # JSON 配置文件路径
    config = load_config(config_file)
    image_dir = config['one_way']['image_dir']
    annotation_dir = config['one_way']['annotation_dir']
    method = config['one_way']['method']
    # image_dir = 'Data/Image'
    # image_dir = 'Data/PIX_feature'
    # annotation_dir = 'Data/Annotation'
    for filename in os.listdir(image_dir):
            if filename.endswith('.png'):
                image_path = os.path.join(image_dir, filename)
                annotation_path = os.path.join(annotation_dir, filename)
                if os.path.exists(annotation_path):
                    print(f'Processing image: {image_path}')
                    if method == 'basic':
                        print('基础算法大津阈值法和形态学处理')
                        basic_process_image(image_path, annotation_path)
                    elif method == 'clustering':
                        print('聚类算法K-means和GMM')
                        Clustering_process_image(image_path, annotation_path)
                    elif method == 'active':
                        print('主动轮廓分割')
                        active_contour_process_image(image_path, annotation_path)
                    elif method == 'graph':
                        print('图割分割') # 仅支持RGB图像
                        graph_process_image(image_path, annotation_path)
                    else:
                        raise ValueError(f"Unknown segmentation method: {method}")

def save_metrics_to_csv(csv_file, metrics):
    df = pd.DataFrame(metrics, index=['Otsu Segmentation', 'Morphological Otsu Segmentation', 'K-means Segmentation', 'GMM Segmentation', 'Active Contour Segmentation', 'Graph Cut Segmentation'],
                      columns=['Dice', 'Sensitivity', 'Specificity', 'Accuracy', 'AUC'])
    df.to_csv(csv_file)

def all_way():
    config_file = 'config.json'  # JSON 配置文件路径
    config = load_config(config_file)
    # image_path = 'Data\Image\ISIC_0000000.png'  # 未提取特征原图
    # image_path = 'Data\PIX_feature\ISIC_0000000.png' # 基于强度特征的图像
    # image_path = 'Data\Texture_feature\ISIC_0000000.png' # 基于纹理特征的图像
    # annotation_path = 'Data\Annotation\ISIC_0000000.png'  # 替换为实际路径  
    image_path = config['all_way']['image_path']
    annotation_path = config['all_way']['annotation_path']

    s1,d1,s2,d2 = basic_process_image(image_path, annotation_path)
    s3,d3,s4,d4 = Clustering_process_image(image_path, annotation_path)
    s5,d5 = active_contour_process_image(image_path, annotation_path)
    s6,d6 = graph_process_image(image_path, annotation_path)

    # 记录评判指标
    # csv_file = 'texture_freature_segmentation_metrics.csv'  # CSV 文件路径
    csv_file = config['all_way']['csv_file']
    metrics = [d1, d2, d3, d4, d5, d6]
    save_metrics_to_csv(csv_file, metrics)

    # 获取文件名和目录名
    file_name = os.path.basename(image_path)
    dir_name = os.path.basename(os.path.dirname(image_path))
    window_title = f"{dir_name}/{file_name}"

    import matplotlib.pyplot as plt
    # 读取原始图像和标注
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    annotation = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)

    # 创建一个图形窗口
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))  # 缩小图像尺寸
    axes = axes.flatten()

    # 设置窗口标题
    fig.suptitle(window_title, fontsize=16)
    # 显示原始图像和标注
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(annotation, cmap='gray')
    axes[1].set_title('Annotation')
    axes[1].axis('off')

    # 显示分割结果
    axes[2].imshow(s1, cmap='gray')
    axes[2].set_title('Otsu Segmentation')
    axes[2].axis('off')

    axes[3].imshow(s2, cmap='gray')
    axes[3].set_title('Morphological Otsu Segmentation')
    axes[3].axis('off')

    axes[4].imshow(s3, cmap='gray')
    axes[4].set_title('K-means Segmentation')
    axes[4].axis('off')

    axes[5].imshow(s4, cmap='gray')
    axes[5].set_title('GMM Segmentation')
    axes[5].axis('off')

    axes[6].imshow(s5, cmap='gray')
    axes[6].set_title('Active Contour Segmentation')
    axes[6].axis('off')

    axes[7].imshow(s6, cmap='gray')
    axes[7].set_title('Graph Cut Segmentation')
    axes[7].axis('off')

    # 显示图形
    plt.tight_layout()
    plt.show()


def get_features():
    # image_dir = 'Data/Image'
    # annotation_dir = 'Data/Annotation'
    config_file = 'config.json'  # JSON 配置文件路径
    config = load_config(config_file)
    image_dir = config['feature']['image_dir']
    annotation_dir = config['feature']['annotation_dir']
    method = config['feature']['method']

    for filename in os.listdir(image_dir):
            if filename.endswith('.png'):
                image_path = os.path.join(image_dir, filename)
                annotation_path = os.path.join(annotation_dir, filename)
                if os.path.exists(annotation_path):
                    if method == 'gabor':
                        get_texture_feature(image_path, annotation_path)
                        print('gabor提取纹理特征')
                    elif method == 'equal':
                        get_pix_feature(image_path, annotation_path)
                        print('自适应直方图均衡化提取强度特征')
                    elif method == 'laws':
                        get_Laws_feature(image_path, annotation_path)
                        print('Laws提取纹理特征')
                    else:
                        raise ValueError(f"Unknown feature method: {method}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='split image')
    parser.add_argument('--show', action='store_true', help='Whether to show generated images')
    parser.add_argument('--remark', action='store_true', help='Whether to show remark of segment images')
    parser.add_argument('--one_way', action='store_true', help='Whether to one way to segemnt all_images')
    parser.add_argument('--all_way',action='store_true',help='Whether to all way to segment one_images')
    parser.add_argument('--get_feature',action='store_true',help='get feature of all_images')
    args = parser.parse_args()

    if args.one_way == True:
        args.show = True
        args.remark = True
        one_way()

    if args.all_way == True:
        all_way()

    if args.get_feature == True:
        get_features()

