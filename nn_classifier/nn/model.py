import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            # nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16)
        )
        conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            #nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )
        flatten = Flatten()
        linear1 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU()
        )
        linear2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )
        linear3 = nn.Sequential(
            nn.Linear(128, 10),
            nn.ReLU()
        )
        
        self.blocks = nn.ModuleList([
            conv1,
            conv2,
            conv3,
            flatten,
            linear1,
            linear2,
            linear3
        ])
    
    
    def forward(self, x):
        
        features = x
        for block in self.blocks:
            features = block(features)
            
        return features

