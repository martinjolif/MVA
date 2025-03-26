
#
#
#      0===========================================================0
#      |       TP6 PointNet for point cloud classification         |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Jean-Emmanuel DESCHAUD - 21/02/2023
#

import numpy as np
import random
import math
import os
import time
import torch
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import sys
import matplotlib.pyplot as plt
# Import functions to read and write ply files
from ply import write_ply, read_ply



class RandomRotation_z(object):
    def __call__(self, pointcloud):
        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),      0],
                               [ math.sin(theta),  math.cos(theta),      0],
                               [0,                               0,      1]])
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return rot_pointcloud

class RandomRotation_y(object):
    def __call__(self, point_cloud):
        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[math.cos(theta), 0, math.sin(theta)],
                               [0, 1, 0],
                               [-math.sin(theta), 0, math.cos(theta)]])
        rot_pointcloud = rot_matrix.dot(point_cloud.T).T
        return rot_pointcloud

class RandomRotation_x(object):
    def __call__(self, point_cloud):
        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[1, 0, 0],
                               [0, math.cos(theta), -math.sin(theta)],
                               [0, math.sin(theta), math.cos(theta)]])
        rot_pointcloud = rot_matrix.dot(point_cloud.T).T
        return rot_pointcloud


class RandomNoise(object):
    def __call__(self, pointcloud):
        noise = np.random.normal(0, 0.02, (pointcloud.shape))
        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud
        

        
class ToTensor(object):
    def __call__(self, pointcloud):
        return torch.from_numpy(pointcloud)


def default_transforms():
    return transforms.Compose([RandomRotation_z(),RandomNoise(),ToTensor()])

def custom_transforms():
    return transforms.Compose([RandomRotation_z(),RandomRotation_y(),RandomRotation_x(),RandomNoise(),ToTensor()])

def test_transforms():
    return transforms.Compose([ToTensor()])



class PointCloudData_RAM(Dataset):
    def __init__(self, root_dir, folder="train", transform=default_transforms()):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir+"/"+dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform
        self.data = []
        for category in self.classes.keys():
            new_dir = root_dir+"/"+category+"/"+folder
            for file in os.listdir(new_dir):
                if file.endswith('.ply'):
                    ply_path = new_dir+"/"+file
                    data = read_ply(ply_path)
                    sample = {}
                    sample['pointcloud'] = np.vstack((data['x'], data['y'], data['z'])).T
                    sample['category'] = self.classes[category]
                    self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pointcloud = self.transforms(self.data[idx]['pointcloud'])
        return {'pointcloud': pointcloud, 'category': self.data[idx]['category']}



class MLP(nn.Module):
    def __init__(self, classes = 10):
        # YOUR CODE
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(3072,512)
        self.batch_norm1 = nn.BatchNorm1d(512)

        self.layer2 = nn.Linear(512,256)
        self.batch_norm2 = nn.BatchNorm1d(256)

        self.layer3 = nn.Linear(256, classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, input):
        # YOUR CODE
        x = input.flatten(start_dim=1)
        x = self.relu(self.batch_norm1(self.layer1(x)))
        x = self.dropout(self.relu(self.batch_norm2(self.layer2(x))))
        x = self.layer3(x)
        return x



class PointNetBasic(nn.Module):
    def __init__(self, classes = 10):
        super().__init__()
        # YOUR CODE
        self.embedding = nn.Sequential(
                                nn.Conv1d(3, 64, 1),
                                nn.BatchNorm1d(64),
                                nn.ReLU(),
                                nn.Conv1d(64, 64, 1),
                                nn.BatchNorm1d(64),
                                nn.ReLU(),
                                nn.Conv1d(64, 64, 1),
                                nn.BatchNorm1d(64),
                                nn.ReLU(),
                                nn.Conv1d(64, 128, 1),
                                nn.BatchNorm1d(128),
                                nn.ReLU(),
                                nn.Conv1d(128, 1024, 1),
                                nn.BatchNorm1d(1024),
                                nn.ReLU(),
                                nn.MaxPool1d(1024, 1))

        self.classifier = nn.Sequential(
                                nn.Linear(1024, 512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(),
                                nn.Linear(512, 256),
                                nn.BatchNorm1d(256),
                                nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(256, classes))


    def forward(self, input):
        # YOUR CODE
        x = self.embedding(input)
        x = x.squeeze(dim=2)
        return self.classifier(x)
        

class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        # YOUR CODE
        self.block1 = nn.Sequential(
            nn.Conv1d(k, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.MaxPool1d(1024, 1)
        )
        self.block2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k**2)
        )

        self.k = k

    def forward(self, input):
        # YOUR CODE
        x = self.block1(input)
        x = self.block2(x.squeeze(dim=2))
        return x.reshape(-1, self.k, self.k) + torch.eye(self.k).repeat(x.shape[0], 1, 1).to(device)


class PointNetFull(nn.Module):
    def __init__(self, classes = 10):
        super().__init__()
        # YOUR CODE
        self.block1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            )

        self.block2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.MaxPool1d(1024, 1))

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, classes))

        self.TNet1 = Tnet(3)
        self.TNet2 = Tnet(64)

    def forward(self, input):
        # YOUR CODE
        rotation_matrix_3x3 = self.TNet1(input)
        x = rotation_matrix_3x3 @ input
        x = self.block1(x)
        rotation_matrix_64x64 = self.TNet2(x)
        x = rotation_matrix_64x64 @ x
        x = self.block2(x)
        x = x.squeeze(dim=2)
        return self.classifier(x), rotation_matrix_3x3



def basic_loss(outputs, labels):
    criterion = torch.nn.CrossEntropyLoss()
    return criterion(outputs, labels)

def pointnet_full_loss(outputs, labels, m3x3, alpha = 0.001):
    criterion = torch.nn.CrossEntropyLoss()
    bs=outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1).to(device)
    if outputs.is_cuda:
        id3x3=id3x3.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)) / float(bs)



def train(model, device, train_loader, test_loader=None, epochs=250):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss=0

    test_accuracy = []
    train_loss = []
    for epoch in range(epochs):
        model.train()
        for i, data in enumerate(train_loader, 0):
            #inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            inputs, labels = data['pointcloud'].to(torch.float32).to(device), data['category'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs.transpose(1,2))
            #outputs, m3x3 = model(inputs.transpose(1,2))
            loss = basic_loss(outputs, labels)
            #loss = pointnet_full_loss(outputs, labels, m3x3)
            loss.backward()
            optimizer.step()
        scheduler.step()
        train_loss.append(loss.item())

        model.eval()
        correct = total = 0
        test_acc = 0
        if test_loader:
            with torch.no_grad():
                for data in test_loader:
                    #inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    inputs, labels = data['pointcloud'].to(torch.float32).to(device), data['category'].to(device)
                    outputs = model(inputs.transpose(1,2))
                    #outputs, __ = model(inputs.transpose(1,2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            test_acc = 100. * correct / total
            test_accuracy.append(test_acc)
            print('Epoch: %d, Loss: %.3f, Test accuracy: %.1f %%' %(epoch+1, loss, test_acc))

    plt.figure()
    plt.plot(train_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.savefig("loss.png")
    plt.show()

    plt.figure()
    plt.plot(test_accuracy)
    plt.xlabel('Epoch')
    plt.ylabel('Test accuracy')
    plt.title('Accuracy Curve')
    plt.savefig("accuracy.png")
    plt.show()

 
if __name__ == '__main__':
    
    t0 = time.time()
    
    ROOT_DIR = "../data/ModelNet10_PLY"
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device: ", device)
    
    train_ds = PointCloudData_RAM(ROOT_DIR, folder='train', transform=custom_transforms())
    test_ds = PointCloudData_RAM(ROOT_DIR, folder='test', transform=test_transforms())

    inv_classes = {i: cat for cat, i in train_ds.classes.items()}
    print("Classes: ", inv_classes)
    print('Train dataset size: ', len(train_ds))
    print('Test dataset size: ', len(test_ds))
    print('Number of classes: ', len(train_ds.classes))
    print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size())

    train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=32)

    #model = MLP(10)
    model = PointNetBasic(10)
    #model = PointNetFull(10)
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print("Number of parameters in the Neural Networks: ", sum([np.prod(p.size()) for p in model_parameters]))
    model.to(device)
    
    train(model, device, train_loader, test_loader, epochs = 150)
    
    t1 = time.time()
    print("Total time for training : ", t1-t0)

    
    


