"""
Custom tattoo matching dataset built from celebrity tattoo images.
Folder structure: data/raw/tattoo_eval/[tattoo_name]/photo_N.jpg
"""
from pathlib import Path
from itertools import combinations
import random
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "tattoo_eval"


class TattooMatchingDataset(Dataset):
    """
    Generates matching pairs from organized tattoo photo folders.
    Returns (image1, image2, is_same) triples.
    """
    
    def __init__(self, data_dir=None, seed=42, max_negative_pairs_per_positive=5):
        """
        Args:
            data_dir: path to the tattoo_eval folder
            seed: random seed for reproducible pair sampling
            max_negative_pairs_per_positive: balance dataset by limiting negatives
        """
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset not found at {self.data_dir}")
        
        self.tattoo_folders = sorted([
            f for f in self.data_dir.iterdir() 
            if f.is_dir() and not f.name.startswith('.')
        ])
        
        if len(self.tattoo_folders) == 0:
            raise ValueError(f"No tattoo folders found in {self.data_dir}")
        
        # collect all images per identity
        self.identity_to_images = {}
        for folder in self.tattoo_folders:
            images = sorted([
                f for f in folder.iterdir() 
                if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}
            ])
            if len(images) >= 2:  # need at least 2 for pairs
                self.identity_to_images[folder.name] = images
        
        # generate pairs
        random.seed(seed)
        self.pairs = self._generate_pairs(max_negative_pairs_per_positive)
    
    def _generate_pairs(self, max_neg_per_pos):
        """Generate balanced positive and negative pairs."""
        positive_pairs = []
        for identity, images in self.identity_to_images.items():
            for img1, img2 in combinations(images, 2):
                positive_pairs.append((img1, img2, 1))
        
        # generate negative pairs (different identities)
        identities = list(self.identity_to_images.keys())
        max_negatives = len(positive_pairs) * max_neg_per_pos
        
        negative_pairs = []
        seen = set()
        attempts = 0
        max_attempts = max_negatives * 10
        
        while len(negative_pairs) < max_negatives and attempts < max_attempts:
            id1, id2 = random.sample(identities, 2)
            img1 = random.choice(self.identity_to_images[id1])
            img2 = random.choice(self.identity_to_images[id2])
            
            pair_key = tuple(sorted([str(img1), str(img2)]))
            if pair_key not in seen:
                seen.add(pair_key)
                negative_pairs.append((img1, img2, 0))
            attempts += 1
        
        all_pairs = positive_pairs + negative_pairs
        random.shuffle(all_pairs)
        return all_pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img1_path, img2_path, is_same = self.pairs[idx]
        
        img1 = np.array(Image.open(img1_path).convert('RGB'))
        img2 = np.array(Image.open(img2_path).convert('RGB'))
        
        return img1, img2, is_same
    
    def stats(self):
        """Print dataset statistics."""
        n_pos = sum(1 for _, _, label in self.pairs if label == 1)
        n_neg = sum(1 for _, _, label in self.pairs if label == 0)
        n_identities = len(self.identity_to_images)
        n_images = sum(len(imgs) for imgs in self.identity_to_images.values())
        
        print(f"Tattoo Matching Dataset Statistics:")
        print(f"  Unique tattoos: {n_identities}")
        print(f"  Total images: {n_images}")
        print(f"  Positive pairs: {n_pos}")
        print(f"  Negative pairs: {n_neg}")
        print(f"  Total pairs: {len(self.pairs)}")
        print(f"\n  Images per tattoo:")
        for identity, images in sorted(self.identity_to_images.items()):
            print(f"    {identity}: {len(images)} images")


if __name__ == "__main__":
    dataset = TattooMatchingDataset()
    dataset.stats()
    
    print("\nSample pair (index 0):")
    img1, img2, is_same = dataset[0]
    print(f"  Image 1 shape: {img1.shape}")
    print(f"  Image 2 shape: {img2.shape}")
    print(f"  Same tattoo: {bool(is_same)}")