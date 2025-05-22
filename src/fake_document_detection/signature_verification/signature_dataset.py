import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SignatureDataset(Dataset):
    def __init__(self, data_path):
        self.pairs = []
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((105, 105)),
            transforms.ToTensor()
        ])

        # Read all person folders
        for person_folder in os.listdir(data_path):
            person_path = os.path.join(data_path, person_folder)
            if not os.path.isdir(person_path):
                continue

            genuine = sorted([f for f in os.listdir(person_path) if f.startswith('org')])
            forged = sorted([f for f in os.listdir(person_path) if f.startswith('for')])

            # Create genuine-genuine (label 0) pairs
            for i in range(len(genuine)):
                for j in range(i + 1, len(genuine)):
                    self.pairs.append((
                        os.path.join(person_path, genuine[i]),
                        os.path.join(person_path, genuine[j]),
                        0
                    ))

            # Create genuine-forged (label 1) pairs
            for g in genuine:
                for f in forged:
                    self.pairs.append((
                        os.path.join(person_path, g),
                        os.path.join(person_path, f),
                        1
                    ))

    def __getitem__(self, index):
        img1_path, img2_path, label = self.pairs[index]
        img1 = self.transform(Image.open(img1_path).convert("L"))
        img2 = self.transform(Image.open(img2_path).convert("L"))
        return img1, img2, torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.pairs)
