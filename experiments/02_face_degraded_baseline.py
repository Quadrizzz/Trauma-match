"""
Baseline evaluation — pretrained ArcFace on DEGRADED LFW test pairs.
Shows how much performance drops under mass casualty conditions.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
from tqdm import tqdm
from src.data.datasets import LFWPairsDataset
from src.features.face_extractor import FaceExtractor
from src.data.augmentation import apply_medical_degradation
from src.utils.metrics import evaluate_pairs


def run_degraded_baseline(severity='medium', degrade_both=True):
    """
    Args:
        severity: 'light', 'medium', or 'heavy'
        degrade_both: if True, degrade both images in the pair.
            If False, only degrade the second image (more realistic — 
            family has clean reference, hospital has degraded photo).
    """
    print(f"Running baseline on {severity} degradation...")
    print(f"Degrade both images: {degrade_both}")
    
    np.random.seed(42)  # reproducible
    
    extractor = FaceExtractor()
    pairs = LFWPairsDataset(subset='test')
    
    similarities = []
    labels = []
    skipped = 0
    
    for i in tqdm(range(len(pairs)), desc=f"{severity} degradation"):
        img1, img2, is_same = pairs[i]
        
        # apply degradation
        if degrade_both:
            img1 = apply_medical_degradation(img1, severity)
        img2 = apply_medical_degradation(img2, severity)
        
        emb1 = extractor.extract(img1)
        emb2 = extractor.extract(img2)
        
        if emb1 is None or emb2 is None:
            skipped += 1
            continue
        
        sim = extractor.similarity(emb1, emb2)
        similarities.append(sim)
        labels.append(is_same)
    
    print(f"\nSkipped {skipped} pairs (face not detected)")
    print(f"Evaluated {len(similarities)} pairs")
    
    metrics = evaluate_pairs(similarities, labels, n_skipped=skipped)
    metrics['severity'] = severity
    metrics['degrade_both'] = degrade_both
    metrics['skipped'] = skipped
    
    print(f"\n=== {severity.upper()} DEGRADATION RESULTS ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    return metrics


def main():
    results_dir = Path(__file__).resolve().parents[1] / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # both images degraded (worst case)
    for severity in ['light', 'medium', 'heavy']:
        key = f"{severity}_both_degraded"
        all_results[key] = run_degraded_baseline(severity, degrade_both=True)
    
    # only one image degraded (realistic mass casualty scenario)
    for severity in ['light', 'medium', 'heavy']:
        key = f"{severity}_one_degraded"
        all_results[key] = run_degraded_baseline(severity, degrade_both=False)
    
    with open(results_dir / "02_face_degraded_baseline.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # summary table
    print("\n\n=== SUMMARY ===")
    print(f"{'Condition':<30} {'Acc':>8} {'EffAcc':>8} {'AUC':>8} {'Det Rate':>10}")
    print("-" * 70)
    for key, metrics in all_results.items():
        print(f"{key:<30} {metrics['accuracy']:>8.4f} {metrics['effective_accuracy']:>8.4f} {metrics['auc']:>8.4f} {metrics['detection_rate']:>10.4f}")
    
    print(f"\nResults saved to {results_dir / '02_face_degraded_baseline.json'}")


if __name__ == "__main__":
    main()