# Trauma-Match

A multimodal biometric identification system designed for mass casualty victim identification, combining facial recognition with detection of distinguishing physical markers (tattoos, piercings, birthmarks, scars, moles) to enable identification when face recognition alone is insufficient.

## The Problem

In mass casualty events — building collapses, terrorist attacks, natural disasters, stampedes — hospitals receive large numbers of unidentified or unconscious patients simultaneously. Families desperately searching for their relatives currently rely on manual processes: phone calls, physical hospital visits, and paper lists. There is no centralized, real-time system that can match a family's photo of their relative to the hospital's photo of an unidentified patient.

Existing systems (e.g., the US National Library of Medicine's People Locator) rely primarily on face matching. However, our experiments show that pretrained state-of-the-art face recognition models fail at face detection for over 69% of severely degraded mass casualty images. When face detection fails, face matching cannot occur. This is a critical failure mode that no existing system explicitly addresses.

## The Solution

Trauma-Match combines multiple complementary identification signals:

1. **Face matching** — when face is detectable
2. **Body marker detection and matching** — tattoos, piercings, birthmarks, scars, moles, and other distinguishing physical features
3. **Physical descriptors** — estimated age, gender, skin tone, build

By combining these signals through late fusion with learned weights, the system can identify victims even when any single modality fails. The hospital staff uploads photos and observed markers; families upload reference photos and describe known distinguishing features. The matching engine produces ranked candidates with confidence scores for human review before any match is communicated.

## Research Contribution

This project produces a research paper with the following novel contributions:

1. **Quantification of face detection failure** in mass casualty conditions through systematic evaluation on synthetically degraded face images
2. **Multimodal architecture** that combines face matching with body marker detection, explicitly designed to handle face detection failure
3. **Broader marker taxonomy** beyond tattoos, recognizing that distinguishing physical features vary across populations
4. **Late fusion evaluation** demonstrating when each modality contributes to successful identification

Target venues: BTAS (IEEE Biometrics: Theory, Applications and Systems), ICB (International Conference on Biometrics), or CoNIMS.

## System Architecture

Hospital photo of unidentified victim
↓
┌───────────────────┐
│  Preprocessing    │
└───────────────────┘
↓
┌──────┬──────┬──────┐
│ Face │ Body │ Desc │
│Model │Marker│Model │
│      │Model │      │
└──────┴──────┴──────┘
↓
Three embedding vectors stored in vector DB
linked to patient record
— later when family searches —
Family photo + descriptions of distinguishing markers
↓
Same preprocessing + three models
↓
Three query vectors
↓
Vector similarity search against DB
↓
Candidates returned with individual scores
↓
Late fusion layer combines scores
↓
Ranked list of matches with confidence
↓
Human review queue → confirmed match → notify family

## Datasets

The project uses publicly available datasets to enable reproducible research:

**Face matching:**
- **Labeled Faces in the Wild (LFW)** — 13,233 images of 5,749 individuals. Standard benchmark for face verification. Used with the official LFW test pairs protocol.
- **CASIA-WebFace** (planned for Month 3) — 494,414 images of 10,575 identities. Used for fine-tuning experiments at scale.

**Body marker detection:**
- **NIST Tatt-C** — Standard tattoo matching dataset used by law enforcement research community. Access requested through NIST.
- **DermaMNIST / ISIC** — Dermatology datasets for skin marker detection (birthmarks, moles).
- **Custom annotations** — Piercings and scars annotated on subsets of LFW.

**Synthetic degradation:**
- Custom augmentation pipeline simulating realistic mass casualty conditions including medical occlusions (oxygen masks, bandages, cervical collars), motion blur, poor lighting, blood and bruising, dust and debris, facial swelling, and wet reflections from sweat or fresh blood.

## Project Flow

The project follows a four-month timeline divided into three months of research and one month of paper writing:

**Month 1 — Face matching baseline (completed)**
- Establish baseline performance of ArcFace on clean and degraded face pairs
- Build and validate synthetic degradation pipeline simulating mass casualty conditions
- Quantify the face detection failure mode under degradation

### Month 2 — Marker Detection and Matching (complete)
Key findings:
| Method | Detection Rate | Accuracy | AUC |
|--------|---------------|----------|-----|
| YOLO (Roboflow trained) | 0% on celebrity images | N/A | N/A |
| Grounding DINO (per-class) | 98.3% | 86.3% (det+CLIP) | 0.796 |
| Whole-image CLIP (baseline) | 100% | 89.9% | 0.884 |

YOLO failed to generalize from the small Roboflow training set. Grounding DINO with per-class text prompts achieves robust marker detection across 5 marker types (tattoo, piercing, scar, birthmark, mole) without any training.

**Month 3 — Multimodal fusion and full evaluation**
- Implement late fusion layer with learned weights
- Train fusion across face, marker, and descriptor modalities
- Run all four experiments (face baseline, marker standalone, fusion, degraded condition analysis)
- Generate final results, figures, and tables

**Month 4 — Paper writing and demo**
- Write paper draft following standard structure (intro, related work, methodology, experiments, results, discussion, conclusion)
- Build minimal demo interface (hospital upload, family search)
- Deploy live demo for portfolio
- Prepare submission for target venue

## Experiments

Four experiments structure the research:

1. **Face matching baseline** — How does face recognition degrade under mass casualty conditions?
2. **Body marker matching standalone** — Can markers identify individuals when face is unavailable?
3. **Multimodal fusion** — Does combining modalities outperform any single one?
4. **Failure mode recovery** — When face detection fails, does the multimodal system recover identification?

## Current Status

Month 1 complete. Key findings:

| Condition | Effective Accuracy |
|-----------|-------------------|
| Clean baseline | 99.6% |
| Light degradation (both images) | 91.6% |
| Medium degradation (both images) | 90.7% |
| Heavy degradation (both images) | 28.5% |
| Heavy degradation (one image) | 51.9% |

Face detection failure rate on heavy degradation: 69%. This quantified failure mode motivates the multimodal approach being developed in Months 2-3.

## Repository Structure

trauma-match/
├── data/                     # datasets (gitignored)
│   ├── raw/                  # downloaded as-is
│   ├── processed/            # preprocessed/augmented
│   └── splits/               # train/val/test definitions
├── src/                      # source code
│   ├── data/                 # data loading and augmentation
│   ├── features/             # feature extraction (face, markers)
│   ├── matching/             # matching engine
│   ├── fusion/               # late fusion layer
│   ├── models/               # model definitions
│   └── utils/                # metrics, visualization
├── notebooks/                # Jupyter/Colab notebooks
├── experiments/              # experiment scripts and results
├── checkpoints/              # trained model weights (gitignored)
├── paper/                    # paper draft and figures
├── api/                      # FastAPI matching engine (future)
└── tests/                    # unit tests

## Setup

Local development environment:

```bash
git clone https://github.com/yourusername/trauma-match.git
cd trauma-match

python -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

Download datasets:

```bash
python -m src.data.download
```

Run baseline experiments:

```bash
python experiments/01_face_baseline.py
python experiments/02_face_degraded_baseline.py
```

## Tech Stack

- **PyTorch** — model training and fine-tuning
- **InsightFace** — pretrained ArcFace for face embedding
- **Albumentations** — synthetic degradation pipeline
- **scikit-learn** — dataset utilities, evaluation metrics
- **YOLOv8** (Ultralytics) — marker detection (Month 2)
- **OpenCV** — image processing
- **Weights & Biases** — experiment tracking
- **FastAPI** — matching engine API (Month 4)

## Ethical Considerations

This project handles sensitive scenarios involving vulnerable individuals. Key ethical principles:

- **Human review required** — No match is communicated to families without human verification
- **Privacy protection** — All biometric data must be encrypted, access-controlled, and deleted after defined retention periods
- **Demographic fairness** — Models will be evaluated across demographic groups to identify and address performance disparities
- **Limitation disclosure** — Results are validated on synthetically degraded data, not real victim photographs. The system is positioned as a feasibility study, not deployment-ready

## Limitations

- Synthetic degradations cannot fully replicate real mass casualty photo conditions
- Datasets used (LFW) have known demographic imbalances that may affect generalization
- The system is a research prototype, not validated for clinical or emergency deployment
- Real-world deployment would require partnerships with disaster response organizations, regulatory approval, and extensive field testing

## Researcher

Abdulmalik Quadri  
Software Engineer at Adzymic | BSc Computer Science, University of Lagos  
Previous research: Published work on facial recognition for forensic analysis at CoNIMS 2024 and CoNIMS 2025 with Prof. Florence Alaba Oladeji

## License

This research code is released under the MIT License. Refer to the LICENSE file for details.

## Citation

If this work is referenced in academic publications, please cite (paper submission in progress):

Quadri, A. (2026). Multimodal Biometric Matching for Mass Casualty Victim Identification:
A Late Fusion Approach Combining Facial Recognition and Body Marking Detection.
[Manuscript in preparation]

