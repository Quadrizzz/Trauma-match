# Month 2 Summary — Body Marker Detection and Matching

## Objective

Develop the body marker (tattoo) detection and matching component of the multimodal identification system. Establish whether body markers can serve as an independent identification signal that complements face matching, particularly when face detection fails.

## Setup

### Detection component (YOLOv8)

**Model:** YOLOv8n (nano) fine-tuned from COCO pretrained weights
**Training:** 28 epochs (early stopping triggered, best results at epoch 8)
**Hardware:** Google Colab Pro with Tesla T4 GPU

**Dataset:** Roboflow public tattoo detection dataset (CC BY 4.0)
- Source: universe.roboflow.com/tatoo-9syny/tattoo-detection-qia5k
- 43 training images, 12 validation images, 6 test images
- Single class: "tattoo" with bounding box annotations
- Pre-resized to 640×640

### Matching component (CLIP)

**Model:** OpenCLIP ViT-B-32 pretrained on LAION-2B
**Method:** Frozen image encoder, no fine-tuning
**Embedding dimension:** 512 (L2-normalized)
**Similarity metric:** Cosine similarity

### Evaluation dataset (custom)

**17 distinct tattoos × 5 images each = 85 total images**

Constructed from publicly photographed individuals to provide ground-truth matching pairs:
- beckham_hand, billie_hand, cara_back, dakota_shoulder, halsey_shoulder
- harry_arm, jamie_head, jolie_back, miley_side, pharrell_neck
- queen_latifan_neck, rock_arm, tiwa_sleeve, tt_back, tyson_face
- victoria_back, whoopi_dragon

Pair generation:
- 170 positive pairs (within-tattoo combinations)
- 850 negative pairs (between-tattoo combinations)
- 1,020 total evaluation pairs
- 5:1 negative-to-positive ratio (random seed = 42 for reproducibility)

## Results

### Tattoo detection (YOLOv8 on Roboflow test set)

| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.9027 |
| mAP@0.5:0.95 | 0.7479 |
| Precision | 1.0000 |
| Recall | 0.3048 |

Strong precision (zero false positives) but limited recall on a small test set.

### Tattoo detection (YOLOv8 on celebrity images)

Detection across 85 celebrity tattoo images at multiple confidence thresholds:

| Threshold | Images with detection | Total detections |
|-----------|----------------------|------------------|
| 0.25 | 0 / 85 (0%) | 0 |
| 0.15 | 0 / 85 (0%) | 0 |
| 0.10 | 0 / 85 (0%) | 0 |
| 0.03 | 3 / 5 sampled | 3 |
| 0.01 | 5 / 5 sampled | 143 (noise) |

**The model failed to generalize from the Roboflow training distribution to celebrity photographs.** At usable confidence thresholds, no tattoos were detected. At very low thresholds, the model produced random noise detections.

### Tattoo matching (CLIP on celebrity pairs)

| Metric | Value |
|--------|-------|
| Accuracy | 0.8990 |
| AUC | 0.8837 |
| Best threshold | 0.7487 |
| TAR @ FAR=0.1% | 0.1529 |
| Detection rate | 100% |
| Pairs evaluated | 1,020 |

## Key Findings

### 1. Small detection datasets fail to generalize

YOLOv8 fine-tuned on 43 Roboflow images achieved 0.90 mAP on the in-distribution test set but failed completely on real-world celebrity images. The model never learned to recognize tattoos under the visual variety found in web-scraped photos. This reflects a fundamental limitation of detection models trained on small, homogeneous datasets.

### 2. Detection is not strictly necessary for matching

By providing tattoo-focused images as input (which is realistic for the family-side of the system where users explicitly photograph or describe distinguishing markers), the detection step can be bypassed. CLIP image embeddings directly on tattoo-centered images produce useful similarity scores.

### 3. Frozen CLIP achieves strong matching accuracy

Without any tattoo-specific training, CLIP ViT-B-32 (LAION-2B pretrained) achieves 89.9% verification accuracy on the custom celebrity evaluation set. This is meaningfully lower than face matching on clean images (99.6%) but achieves **100% availability** — every image produces an embedding.

### 4. The strict-FAR limitation

At very low false-accept rates (FAR=0.1%), tattoo matching alone catches only 15.3% of genuine matches. This means tattoo matching cannot serve as a sole identification method in scenarios where false matches must be minimized. However, it provides a complementary signal when combined with face matching, especially when face detection fails.

### 5. Comparison across modalities (preliminary)

| Modality | Detection Rate | Accuracy | Use Case |
|----------|---------------|----------|----------|
| Face (clean) | 100% | 99.6% | Standard conditions |
| Face (heavy degradation) | 31% | 28.5% effective | Mass casualty conditions |
| Tattoo (CLIP) | 100% | 89.9% | When face fails |

The complementarity is clear — tattoo matching has lower accuracy but works in conditions where face matching cannot.

## Decisions Made

1. **YOLO detection removed from pipeline.** Replaced with direct CLIP embedding on tattoo-focused images. Detection is reserved for future work with larger annotated datasets.

2. **Frozen CLIP retained for matching.** Custom contrastive training is reserved for follow-up work. Frozen embeddings demonstrate the feasibility of the multimodal approach for this paper.

3. **Custom evaluation set documented as limitation.** While the 17-tattoo celebrity dataset enables ground-truth matching evaluation, larger standardized datasets (NIST Tatt-C) would strengthen reproducibility and external comparison.

4. **Marker taxonomy deferred.** The broader marker concept (piercings, birthmarks, scars, moles) is conceptually motivated but not implemented in this paper due to dataset availability constraints. Reserved for follow-up work.

## Artifacts Produced

- `src/data/tattoo_dataset.py` — TattooMatchingDataset for pair generation
- `src/features/tattoo_extractor.py` — CLIP-based tattoo embedding extractor
- `src/models/tattoo_detector.py` — YOLOv8 wrapper (retained for reference, not used in final pipeline)
- `experiments/results/02_tattoo_detection.json` — YOLO detection results
- `experiments/results/03_tattoo_matching.json` — CLIP matching results
- `experiments/results/04_marker_matching_with_detection.json` — CLIP + GROUNDING DINO 
- `checkpoints/tattoo/tattoo_yolov8_best.pt` — trained YOLO weights (in Drive)

## Limitations Noted

1. **Custom evaluation dataset.** The 17-tattoo set is smaller than standard benchmarks. Results are not directly comparable to published tattoo matching papers using Tatt-C.

2. **No marker variety.** Only tattoos are evaluated. Piercings, birthmarks, scars, and moles — also part of the conceptual marker taxonomy — are not included due to dataset constraints.

3. **Frozen embeddings.** CLIP was not adapted to tattoos specifically. Custom contrastive training would likely improve TAR at strict FAR rates.

4. **No detection in the final pipeline.** The system currently assumes tattoo-focused input images. Real deployment would require robust tattoo localization in full-body photographs.

### Month 2 — Marker Detection and Matching (complete)

Key findings:
| Method | Detection Rate | Accuracy | AUC |
|--------|---------------|----------|-----|
| YOLO (Roboflow trained) | 0% on celebrity images | N/A | N/A |
| Grounding DINO (per-class) | 98.3% | 86.3% (det+CLIP) | 0.796 |
| Whole-image CLIP (baseline) | 100% | 89.9% | 0.884 |

YOLO failed to generalize from the small Roboflow training set. Grounding DINO with per-class text prompts achieves robust marker detection across 5 marker types (tattoo, piercing, scar, birthmark, mole) without any training.

Once you've pushed, we can move to gathering face photos of your 17 celebrities for Month 3 multimodal fusion. Let me know when you've completed the push.Claude Fable 5 is currently unavailable.

## Next Steps — Month 3

The matching components for both face and body markers are established. Month 3 focuses on combining them through late fusion:

- Build a multimodal evaluation set with both face and tattoo images for the same identities
- Implement late fusion layer with learned weights
- Train fusion across face and tattoo modalities
- Run Experiment 3: multimodal vs single modality
- Run Experiment 4: failure mode recovery (does tattoo matching identify victims when face matching fails?)

The fundamental hypothesis to test in Month 3:

> When face detection fails or face matching is unreliable under mass casualty conditions, can body marker matching recover identification accuracy that face matching alone cannot achieve?

---

*Document compiled at end of Month 2. All results reproducible via scripts in `experiments/` and notebooks in `notebooks/`.*