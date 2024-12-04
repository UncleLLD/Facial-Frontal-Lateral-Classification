import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # set GPU
else:
    device = torch.device("cpu")   # set CPU


class ImageDatasetWithLabels(Dataset):
    def __init__(self, folder_dir, transform=None, label_file='labels.txt'):
        self.img_dir = folder_dir
        self.transform = transform
        self.label_file = label_file
        self.image_names = []
        self.labels = []
        with open(label_file, 'r') as f:
            for ln in f:
                image_path = ln.rstrip('\n')
                image_name = image_path.split('/')[-1]
                self.image_names.append(image_path)
                if image_name.startswith('front'):
                    self.labels.append(0)
                else:
                    self.labels.append(1)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

    def split(self, train=True):
        if train:
            return self[:self.train_size]
        return self[self.train_size:]

class FaceSideClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(FaceSideClassifier, self).__init__()
        # load the pre-trained model
        # self.backbone = models.efficientnet_b0(pretrained=True)
        backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Linear(2048, 1024)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.output = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.output(x)
        return x


if __name__ == '__main__':
    transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    folder_dir = '../../front-side-faces/'
    data = ImageDatasetWithLabels(folder_dir, transform=transforms, label_file='labels.txt')
    train_size = int(0.8 * len(data))  # train 80%
    test_size = len(data) - train_size  # test 20%
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])

    batch_size = 64
    num_workers = 4
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, num_workers=num_workers)

    num_epochs = 1
    model = FaceSideClassifier().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    writer = SummaryWriter('logs')

    # train
    for epoch in range(num_epochs):
        running_loss = 0.0
        iter_nums = 0
        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            iter_nums += 1
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if iter_nums % 10 == 9:
                print('[%d, %5d] loss: %.3f' % (epoch+1, iter_nums+1, running_loss/10))
                writer.add_scalar('training_loss', running_loss/10, epoch*len(train_dataloader)+iter_nums)
                running_loss = 0

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    # test
    model.eval()
    total_correct = 0
    total_samples = 0
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
        accuracy = total_correct / total_samples
        average_loss = test_loss / len(test_dataloader)
        writer.add_scalar('test_accuracy', accuracy)
        writer.add_scalar('test_loss', average_loss)
        print(f"accuracy: {accuracy}, loss: {average_loss}")

    writer.close()
    torch.save(model, 'model.pth')
