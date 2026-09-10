import os
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

class FlowerDataset(Dataset):
    def __init__(self, annotations, img_dir, caption_transform=None, image_transform=None):
        super().__init__()
        self.img_labels = pd.read_csv(annotations)
        self.img_dir = img_dir
        self.image_transform = image_transform
        self.caption_transform = caption_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        row = self.img_labels.iloc[index]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        caption = row['caption']

        if self.image_transform:
            image = self.image_transform(image)

        if self.caption_transform:
            caption = self.caption_transform(caption)

        return caption, image