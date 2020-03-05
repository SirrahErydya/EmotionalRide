"""
A deep CNN to classify the emotions in a human face.
Architecture inspired by https://arxiv.org/pdf/1910.05602.pdf
:author: Fenja Kollasch
"""
import torch.nn as nn


class EmotionClassifier(nn.Module):
    def __init__(self):
        super(EmotionClassifier, self).__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(0.25),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1,  padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1), nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1,  padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  nn.Dropout(0.25),
            nn.Flatten()
        )
        self.fc1 = nn.Linear(in_features=20736, out_features=1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.output_layer = nn.Linear(in_features=1024, out_features=7)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.convolutions(x)
        x = self.dropout(self.relu((self.fc1(x))))
        x = self.output_layer(x)
        return self.softmax(x)