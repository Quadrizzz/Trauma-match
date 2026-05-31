"""
Baseline evaluation — pretrained ArcFace on LFW test pairs.
This establishes the number to beat with fine-tuning.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from tqdm import tqdm
from src.data.datasets import LFWPairsDataset
from src.features.face_extractor import FaceExtractor
from src.utils.metrics import evaluate_pairs


def run_baseline():
    print("Loading face extractor (ArcFace buffalo_l)...")
    extractor = FaceExtractor()
    
    print("Loading LFW test pairs...")
    pairs = LFWPairsDataset(subset='test')
    print(f"Evaluating on {len(pairs)} pairs")
    
    similarities = []
    labels = []
    skipped = 0
    
    for i in tqdm(range(len(pairs)), desc="Extracting embeddings"):
        img1, img2, is_same = pairs[i]
        
        emb1 = extractor.extract(img1)
        emb2 = extractor.extract(img2)
        
        if emb1 is None or emb2 is None:
            skipped += 1
            continue
        
        sim = extractor.similarity(emb1, emb2)
        similarities.append(sim)
        labels.append(is_same)
    
    print(f"\nSkipped {skipped} pairs (face not detected)")
    print(f"Evaluated {len(similarities)} pairs successfully")
    
    print("\n=== BASELINE RESULTS ===")
    metrics = evaluate_pairs(similarities, labels)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # save results
    results_dir = Path(__file__).resolve().parents[1] / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(results_dir / "01_face_baseline.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to experiments/results/01_face_baseline.json")
    return metrics


if __name__ == "__main__":
    run_baseline()