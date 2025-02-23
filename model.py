import os

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

from PIL import Image

from mydataset import transform
from config import BASE_DIR


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 112x112
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 56x56
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28x28
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 28 * 28, 512), nn.ReLU(), nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# Define the ViT-based model
class ViTModel(nn.Module):
    def __init__(self):
        super(ViTModel, self).__init__()
        # Load a pre-trained ViT model
        self.vit = models.vit_b_16(pretrained=True)

        # Replace the head for binary classification
        if isinstance(self.vit.heads, nn.Sequential):
            in_features = self.vit.heads[
                0
            ].in_features  # Assuming first layer is Linear
        else:
            in_features = self.vit.heads.in_features
        self.vit.heads = nn.Linear(in_features, 2)  # Binary output

    def forward(self, x):
        x = self.vit(x)
        return x


def load_model(model_name):
    model_path = os.path.join(BASE_DIR, model_name + ".pth")
    if model_name.startswith("vit_"):
        model = ViTModel()
    else:
        model = CNNModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def predict(model, image_path):
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    image = image.to(DEVICE)

    with torch.no_grad():
        output = model(image)
        cls = torch.argmax(F.softmax(output, dim=1), dim=1).item()

    return cls, output
