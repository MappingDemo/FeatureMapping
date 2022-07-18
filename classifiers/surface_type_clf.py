import os
import json

import torch
import torch.nn as nn
import torchvision.transforms.functional as F

from lib.recognition.base_attribute_clf import BaseAttributeClassifier

class CNN(nn.Module):
    def __init__(self, N_classes: int):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2), # kernel sizes 3x3 
            nn.BatchNorm2d(16),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.2),
            nn.ReLU(), 
            nn.MaxPool2d(2),
            nn.Dropout(p=0.2),
        )
        self.fc = nn.Linear(32*32*2, N_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    
class SurfaceTypeClassifier(BaseAttributeClassifier):
    
    def __init__(self, model=CNN, class_names=[0]):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model(len(class_names)).to(self._device)
        self._class_names = class_names

    def __save__(self, PATH: str):
        # Save model weights (recommended method instead for PyTorch, instead of pickling full object)
        torch.save(self._model.state_dict(), os.path.join(PATH, 'model.pth'))
        
    def __load__(self, PATH: str):
        # Load model weights
        self._model = CNN(len(self._class_names)).to(self._device)
        self._model.load_state_dict(torch.load(os.path.join(PATH, 'model.pth')))
        self._model.eval()

    def predict(self, parking_prediction):
        result = {}
        image = parking_prediction.get_squared_image(size=(32,32), cmap='rgb')
        # print(image.shape)
        x = F.to_tensor(image).unsqueeze(0).to(self._device)
        # print(x.shape)
        outs = self._model(x)
        # print(outs)
        class_id = int(torch.argmax(outs))
        # print(class_id)
        result['surface_type'] = self._class_names[class_id]
        return result