import os
import random
from torchvision import transforms
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from qqdm import qqdm
from utils import UNet


# 定义文件夹路径
data_dir = 'Data4AI/Data4CNN/UDIAT'
train_images_dir = os.path.join(data_dir, 'imgs')
train_annotations_dir = os.path.join(data_dir, 'gt')

# 获取图像文件名列表
image_filenames = os.listdir(train_images_dir)
image_filenames.sort()  # 确保图像和标签顺序对应

# 划分数据集：80% 训练集，20% 测试集
train_filenames, test_filenames = train_test_split(image_filenames, test_size=0.2, random_state=42)


def get_image_paths(filenames, images_dir, annotations_dir):
    image_paths = []
    annotation_paths = []
    
    for filename in filenames:
        image_path = os.path.join(images_dir, filename)
        annotation_path = os.path.join(annotations_dir, filename)
        image_paths.append(image_path)
        annotation_paths.append(annotation_path)
        
    return image_paths, annotation_paths

# 获取训练集和测试集的图像及标签路径
train_image_paths, train_annotation_paths = get_image_paths(train_filenames, train_images_dir, train_annotations_dir)
test_image_paths, test_annotation_paths = get_image_paths(test_filenames, train_images_dir, train_annotations_dir)

print('Number of training images:', len(train_image_paths))
print('Number of testing images:', len(test_image_paths))


class UNetDataset(Dataset):
    def __init__(self, image_paths, annotation_paths, transform=None):
        self.image_paths = image_paths
        self.annotation_paths = annotation_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 读取图像和标签
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
        annotation = cv2.imread(self.annotation_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # 转换图像和标签为适合网络输入的形式
        if self.transform:
            image = self.transform(image)
            annotation = self.transform(annotation)
        
        return image, annotation
    

# 定义转换
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


train_dataset = UNetDataset(train_image_paths, train_annotation_paths, transform)
test_dataset = UNetDataset(test_image_paths, test_annotation_paths,transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 创建保存图片的目录
os.makedirs('picsg/image', exist_ok=True)
os.makedirs('picsg/ground_truth', exist_ok=True)
os.makedirs('picsg/predicted', exist_ok=True)
os.makedirs('picsg/check_points', exist_ok=True)


# 初始化模型、损失函数和优化器
# 检查是否有可用的 GPU
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    model.load_state_dict(torch.load('picss/check_points/model_epoch_400.pth'))
    print('Model loaded picss/check_points/model_epoch_400.pth')

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs=200
    for e,epoch in enumerate(range(num_epochs)):
        progress_bar = qqdm(train_loader)
        model.train()
        running_loss = 0.0
        for i,(images, annotations) in enumerate(progress_bar):
                images = images.cuda()
                annotations = annotations.cuda()
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, annotations) # loss已被除以batch_size
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                progress_bar.set_infos({
                    'Loss': round(loss.item(), 4),
                    'Epoch': e+1,
                    })
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Train_Loss: {epoch_loss:.4f}')

        
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, annotations in test_loader:
                images = images.cuda()
                annotations = annotations.cuda()
                outputs = model(images)
                loss = criterion(outputs, annotations)
                running_loss += loss.item() * images.size(0)

            test_loss = running_loss / len(test_loader.dataset)
            print(f'Test_Loss: {test_loss:.4f}')

        
        # 每隔5轮显示测试集中的输入图片和模型的输出图片
        # 每隔5轮显示测试集中的输入图片和模型的输出图片
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                import matplotlib.pyplot as plt
                # 创建一个图形窗口
                fig, axes = plt.subplots(2, 4, figsize=(10, 5))  # 缩小图像尺寸
                axes = axes.flatten()
                for i in range(8):  # 显示8张图片
                    image_path = test_image_paths[i]
                    annotation_path = test_annotation_paths[i]
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    annotation = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)

                    annotation = cv2.resize(annotation, (256, 256))
                    image = cv2.resize(image, (256, 256))
                
                    image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
                    output = model(image_tensor)
                    
                    output = torch.sigmoid(output).squeeze().cpu().numpy()
                    output = (output > 0.5).astype(np.uint8) * 255
                    
                    axes[i].imshow(output, cmap='gray')
                    axes[i].set_title(f'Output {i}')
                    axes[i].axis('off')

                    cv2.imwrite(f'picsg/image/input_image_img_{i}.jpg', image)
                    cv2.imwrite(f'picsg/ground_truth/ground_truth_img_{i}.jpg', annotation)
                    
                
                
                plt.tight_layout()
                plt.savefig(f'picsg/predicted/model_output_epoch_{epoch + 1}.jpg')
                plt.close(fig)

        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), f'picsg/check_points/model_epoch_{epoch + 1}.pth')
            print(f'Model saved at epoch {epoch + 1}')

# 主函数
if __name__ == '__main__':
    train()
