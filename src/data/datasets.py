"""
PyTorch Dataset classes for LFW and other face datasets.
"""
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.datasets import fetch_lfw_people, fetch_lfw_pairs
import numpy as np
import torch

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "lfw" / "lfw_home" / "lfw_funneled"


class LFWPeopleDataset(Dataset):
    """LFW dataset for training — returns (image, identity_label) pairs."""
    
    def __init__(self, min_faces_per_person=20, transform=None):
        self.transform = transform
        
        data = fetch_lfw_people(
            data_home=str(DATA_DIR),
            min_faces_per_person=min_faces_per_person,
            resize=1.0,
            color=True,
            funneled=True,
        )
        
        # images shape: (n_samples, height, width, channels)
        self.images = data.images
        self.labels = data.target
        self.target_names = data.target_names
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # image comes as float [0, 1] from sklearn — convert to uint8 [0, 255]
        image = (self.images[idx] * 255).astype(np.uint8)
        label = int(self.labels[idx])
        
        if self.transform:
            image = self.transform(image=image)["image"]
        
        return image, label
    
    @property
    def num_classes(self):
        return len(self.target_names)


class LFWPairsDataset(Dataset):
    """
    LFW pairs dataset for evaluation.
    Returns (image1, image2, is_same_person) triples.
    """
    
    def __init__(self, subset='test', transform=None):
        self.transform = transform
        
        data = fetch_lfw_pairs(
            data_home=str(DATA_DIR),
            subset=subset,
            resize=1.0,
            color=True,
        )
        
        # pairs shape: (n_pairs, 2, height, width, channels)
        self.pairs = data.pairs
        self.labels = data.target  # 1 = same person, 0 = different
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img1 = (self.pairs[idx][0] * 255).astype(np.uint8)
        img2 = (self.pairs[idx][1] * 255).astype(np.uint8)
        is_same = int(self.labels[idx])
        
        if self.transform:
            img1 = self.transform(image=img1)["image"]
            img2 = self.transform(image=img2)["image"]
        
        return img1, img2, is_same


if __name__ == "__main__":
    # quick sanity check
    print("Loading LFW People...")
    people = LFWPeopleDataset()
    print(f"  Total images: {len(people)}")
    print(f"  Unique identities: {people.num_classes}")
    img, label = people[0]
    print(f"  Sample image shape: {img.shape}, dtype: {img.dtype}")
    print(f"  Sample label: {label} ({people.target_names[label]})")
    
    print("\nLoading LFW Pairs (test set)...")
    pairs = LFWPairsDataset()
    print(f"  Total pairs: {len(pairs)}")
    img1, img2, is_same = pairs[0]
    print(f"  Sample pair shape: {img1.shape}")
    print(f"  Same person: {bool(is_same)}")