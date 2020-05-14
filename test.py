import numpy as np # linear algebra
import shutil
import os

#PyTorch packages
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler

import PIL
from PIL import Image

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

classes = ["Fresh Apple", "Fresh Banana", "Fresh Orange", "Rotten Apple", "Rotten Banana", "Rotten Orange"]
    
model = CNN()

device = torch.device('cpu')
state_dict = torch.load("model/fruitsModel.pt", map_location=device)
model.load_state_dict(state_dict)

images = os.listdir("./testImages/")
for i in range(len(images)):
    images[i] = Image.open("./testImages/" + images[i])

def process_image(image):
    transformations = transforms.Compose(
        [transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])])
    
    image = transformations(image)
    return image

def predict(image, model):
    image = process_image(image)
    image = np.expand_dims(image, 0)
    image = torch.from_numpy(image)

    model.eval()
    output = model(image)
    _, pred = torch.max(output, 1)
    return classes[np.squeeze(pred.cpu().numpy())]

for image in images:
    print(predict(image, model))