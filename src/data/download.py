"""
Download datasets for the matching engine.
"""
from pathlib import Path
from sklearn.datasets import fetch_lfw_people, fetch_lfw_pairs

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def download_lfw():
    """Download LFW via scikit-learn (uses a reliable mirror)."""
    lfw_dir = DATA_DIR / "lfw"
    lfw_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading LFW people...")
    # min_faces_per_person=20 means only include people with at least 20 photos
    # this gives us enough images per identity for evaluation
    people = fetch_lfw_people(
        data_home=str(lfw_dir),
        min_faces_per_person=20,
        resize=1.0,
        color=True,
        funneled=True,
    )
    
    print(f"Loaded {len(people.images)} images of {len(people.target_names)} people")
    
    print("Downloading LFW pairs for evaluation...")
    pairs = fetch_lfw_pairs(
        data_home=str(lfw_dir),
        subset='test',
        resize=1.0,
        color=True,
    )
    
    print(f"Loaded {len(pairs.pairs)} evaluation pairs")
    print(f"LFW data home: {lfw_dir}")
    
    return people, pairs


if __name__ == "__main__":
    download_lfw()