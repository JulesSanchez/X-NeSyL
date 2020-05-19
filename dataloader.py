import torch
from torchvision import datasets, transforms as T
from torch.utils.data import Dataset
from PIL import Image 
import pandas as pd

normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])
val_transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])

class ArchitectureClassificationDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, batch_size, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images_loc = pd.read_csv(csv_file)
        self.transform = transform
        self.batch_size = batch_size

    def __len__(self):
        return len(self.images_loc)//self.batch_size + (len(self.images_loc)%self.batch_size > 0)

    def __getitem__(self, idx):

        imgs = []
        labels = []
        
        for k in range(self.batch_size*idx, min(self.batch_size*(idx+1), len(self.images_loc))):

            img_name = self.images_loc.iloc[k, 0]
            image = Image.open(img_name).convert('RGB')
            labels.append(self.images_loc.iloc[k, 1])
            if self.transform:
                imgs.append(self.transform(image))

        return torch.stack(imgs), torch.LongTensor(labels)