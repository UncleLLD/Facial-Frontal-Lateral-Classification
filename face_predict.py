import os
import sys
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from PIL import Image
from face_classifiy import FaceSideClassifier

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class FacePredict:
    def __init__(self):
        # 定义模型和其他必要的组件
        self.model = torch.load('model.pth')
        self.model.eval()
        self.model = self.model.to(device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def predict(self, image_file):
        image = Image.open(image_file)
        image = self.transform(image)
        image = image.unsqueeze(0)  # 添加一个维度作为批次维度
        image = image.to(device)

        # 使用加载的模型进行预测
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, dim=1)
        return predicted.item()


if __name__ == '__main__':
    image_file = sys.argv[1]
    fp = FacePredict()
    res = fp.predict(image_file)
    print("预测类别:", res)
