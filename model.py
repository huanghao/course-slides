import torch
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
from PIL import Image

MODEL_PATH = "/Users/huanghao/workspace/learning/llm/vids/slide_classifier.pth"
MODEL_PATH = "/Users/huanghao/workspace/learning/llm/vids/slide_classifier_004_20250210_111151.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
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


def load_model():
    model = MyModel()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model


def predict(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    image = image.to(DEVICE)

    with torch.no_grad():
        output = model(image)
        cls = torch.argmax(F.softmax(output, dim=1), dim=1).item()

    return cls, output
