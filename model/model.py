# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import shutil

#PyTorch packages
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler

import helper
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

path = "../input/fruits-fresh-and-rotten-for-classification/dataset"
batch_size = 32
valid_size = 0.2

#Transformations on images
transforms = transforms.Compose([transforms.RandomRotation(30),
                                 transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5, 0.5, 0.5],
                                                      [0.5, 0.5, 0.5])])

train_dataset = datasets.ImageFolder(path + "/train", transform=transforms) #Training
test_dataset = datasets.ImageFolder(path + "/test", transform=transforms) #Testing

#Splitting training dataset for training and validation
num_train = len(train_dataset)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

#Samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
test_dataloader = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, shuffle=True)
validation_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
dataiter = iter(train_dataloader)
images, labels = dataiter.next()

imshow(images[0])
print(labels[0])

classes = os.listdir(path + "/train")

#CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #Conv and pooling layers
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.conv5 = nn.Conv2d(128, 256, 1)
        self.pool = nn.MaxPool2d(2, 2)
        #Linear and dropout laters
        self.f1 = nn.Linear(256*6*6, 512)
        self.f2 = nn.Linear(512, 6)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        #Pooling and conv
        x = self.pool(F.relu(self.conv1(x))) #size (16,110,110)
        x = self.pool(F.relu(self.conv2(x))) #size (32,54,54)
        x = self.pool(F.relu(self.conv3(x))) #size (64,26,26)
        x = self.pool(F.relu(self.conv4(x))) #size (128,12,12)
        x = self.pool(F.relu(self.conv5(x))) #size (256,6,6)
        #Linear and dropout
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        
        x = F.relu(self.f1(x))
        x = self.dropout(x)
        
        x = F.log_softmax(self.f2(x), dim=1)
        return x

    
model = CNN()
print(model)

train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    model.cuda()
    print("GPU is avaliable! Training on GPU ...")
else:
    print("Training on CPU ...")

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 50
valid_loss_min = np.Inf
for i in range(epochs):
    train_loss = 0
    val_loss = 0
    #Training
    model.train()
    for data, labels in train_dataloader:
        if train_on_gpu:
            data, labels = data.cuda(), labels.cuda()
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
    #Evaluating
    with torch.no_grad():
        accuracy = 0
        valid_total = 0
        valid_correct = 0
        
        model.eval()
        for data. labels in validation_dataloader:
            if train_on_gpu:
                data, labels = data.cuda(), labels.cuda()

            output = model(data)
            loss = criterion(output, labels)
            val_loss += loss.item()*data.size(0)
            
            scores, predictions = torch.max(output.data, 1)
            valid_total += labels.size(0)
            valid_correct += int(sum(predictions == labels))
            
        acc = round((valid_correct / valid_total), 2)

        train_loss /= len(train_dataloader.sampler)
        val_loss /= len(validation_dataloader.sampler)

        print("Epoch:", i+1, "\t Training Loss:", train_loss, "\t Validation Loss:", val_loss, "\t Accuracy:", acc)
        if val_loss < valid_loss_min:
            print("Validation Loss decreased from", valid_loss_min, "to", val_loss)
            print("Model saved!")
            torch.save(model.state_dict(), "fruitsModel.pt")
            valid_loss_min = val_loss