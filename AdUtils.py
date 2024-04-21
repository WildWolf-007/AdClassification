import torch
import torch.nn as nn
from torchvision.transforms import transforms
from PIL import Image

class AdClassification(nn.Module):

    def __init__(self, input_shape : int, hidden_shape:int, output_shape : int) -> None:
        super(AdClassification, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_shape, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_shape, out_channels=hidden_shape, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_shape, out_channels=hidden_shape, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_shape, out_channels=hidden_shape, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_shape*62*62, out_features=output_shape)
        )

    def forward(self, x:torch.Tensor):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.classifier(x2)
        return x3
    


transform = transforms.Compose(
    [transforms.Resize(size = (250, 250)),
    transforms.RandomRotation(degrees= 45),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.RandomVerticalFlip(p = 0.5),
    transforms.ToTensor()])

def transform_image(image_bytes):
    image = Image.open(image_bytes)
    return transform(image)