import os

from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms

from config import IMAGE_PATH, POSITIVE_LABEL_FILE, NEGATIVE_LABEL_FILE

IMG_SIZE = 224

transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),  # 调整到相同尺寸
        transforms.ToTensor(),  # 转换为 PyTorch 张量
        transforms.Normalize([0.5], [0.5]),  # 归一化
    ]
)


class CustomDataset(Dataset):
    def __init__(self, pos_file, neg_file, transform=None):
        self.data = self._load(pos_file, neg_file)
        self.transform = transform

    def _load(self, pos_file, neg_file):
        positive_images, negative_images = set(), set()

        with open(pos_file, "r") as f:
            positive_images = set(line.strip() for line in f)

        with open(neg_file, "r") as f:
            negative_images = set(line.strip() for line in f)

        xy = [(i, 1) for i in positive_images] + [(i, 0) for i in negative_images]
        return xy

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img_path = os.path.join(IMAGE_PATH, img_path)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def split_dataset(batch_size, train_ratio):
    dataset = CustomDataset(
        pos_file=POSITIVE_LABEL_FILE, neg_file=NEGATIVE_LABEL_FILE, transform=transform
    )
    print("dataset size", len(dataset))
    train_size = int(train_ratio * len(dataset))
    dev_size = len(dataset) - train_size
    print("tr size", train_size, "dev size", dev_size)
    train_dataset, dev_dataset = random_split(dataset, [train_size, dev_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    return train_dataset, dev_dataset, train_loader, dev_loader
