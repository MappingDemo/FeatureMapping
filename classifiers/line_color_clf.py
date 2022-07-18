import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable

import sys,os

from .base_attribute_clf import BaseAttributeClassifier
import torchvision.transforms.functional as F


class ResBlock(nn.Module):
    
    """
    Defines a single ResNet Block which will be used to make the ResNet structure
    """
    
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        out = nn.ELU()(self.bn1(self.conv1(input)))
        out = nn.ELU()(self.bn2(self.conv2(out)))
        out = out + shortcut
        return nn.ELU()(out) 
    

class ResNet(nn.Module):
    
    """
    The ResNet model
    Input:
    in_channels: if RGB images the in_channels is 3. If the image is grayscale the in_channels is 1.
    resblock: takes the Resnet block as input as defined above
    outputs: the number of classes
    """
    
    def __init__(self, in_channels, resblock, outputs=4):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False)
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(256, outputs)

    def forward(self, input): 
        out = self.layer0(input) 
        out = self.layer1(out) 
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.gap(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)

        return out
    

class LineColorClassifier(BaseAttributeClassifier):
    
    
    def __init__(self, model = ResNet, class_names=[]):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._class_names = class_names
        self._model = ResNet(3, ResBlock, outputs=len(self._class_names))

    def __save__(self, PATH: str):

        """
        save model to model.pth
        """
        
        torch.save(self._model.state_dict(), os.path.join(PATH, 'model.pth'))

    def __load__(self, PATH: str):
        
        """
        Used to load the model weights 
        """
        
        self._model = ResNet(3, ResBlock, outputs=len(self._class_names)).to(self._device)
        self._model.load_state_dict(torch.load(os.path.join(PATH, 'model.pth')))
        self._model.eval()

    def predict(self, parking_prediction) -> str:

        """
        input: Instance of lib.detection.prediction.ParkingPrediction
        output: dict of {"line_color": __color__ }
        """

        result = {}
        # get image of the parkign space's body
        body_image = parking_prediction.get_body_image(size=(64, 64), cmap='rgb')
        # convert to tensor
        image_tensor = F.to_tensor(body_image).unsqueeze(0).to(self._device)
        predict = self._model(image_tensor)
        class_id = torch.argmax(predict)
        result['line_color'] = self._class_names[class_id]
        return result
        
        
        
    
    
        