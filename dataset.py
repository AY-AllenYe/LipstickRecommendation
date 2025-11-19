import jittor as jt
from jittor import transform
from jittor.dataset.dataset import Dataset
import os
from PIL import Image
import numpy as np

class LipstickDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.transform = transform
        self.image_labels = []

        with open(label_file, 'r') as f:
            for line in f:
                name, label = line.strip().split()
                self.image_labels.append((name, int(label)))

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, index):
        name, label = self.image_labels[index]
        path = os.path.join(self.image_dir, name)
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, jt.int32(label)


def get_transforms():
    return transform.Compose([
        # transform.Resize((224, 224)),
        # transform.RandomHorizontalFlip(),
        # transform.ImageNormalize(mean=[0.5], std=[0.5]),
        transform.ToTensor()
    ])
