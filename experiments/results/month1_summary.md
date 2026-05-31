# Month 1 Summary — Face Matching Baseline

## Objective

Establish the baseline performance of state-of-the-art face recognition on clean and degraded face images, quantifying the performance gap that motivates the multimodal approach in this project.

## Setup

**Model:** ArcFace via InsightFace (`buffalo_l` model pack, pretrained on Glint360K)  
**Embedding dimension:** 512  
**Detection input size:** 160×160 (tuned for small LFW images)  
**Detection threshold:** 0.3  

**Dataset:** Labeled Faces in the Wild (LFW)
- Training set (`LFWPeopleDataset`): 3,023 images of 62 unique identities (min_faces_per_person=20)
- Evaluation set (`LFWPairsDataset` test subset): 1,000 pairs (500 same-person, 500 different-person)

**Degradation pipeline:** Custom augmentation module simulating realistic mass casualty conditions:
- Region-targeted medical occlusions (oxygen mask, forehead bandage, cheek bandage, cervical collar)
- Motion blur and Gaussian blur (shaky hands, poor focus)
- Brightness reduction (poor lighting in emergency conditions)
- Blood and bruising patches (irregular dark red elliptical shapes)
- Dirt and dust layer (collapse and debris scenarios)
- Facial swelling via elastic transform
- Wet reflections (sweat, fresh blood)

Three severity levels:
- **Light:** clinic conditions, minor injury
- **Medium:** ambulance or ER conditions, moderate trauma
- **Heavy:** disaster scene conditions, severe trauma

## Evaluation Protocol

Standard LFW verification protocol: for each pair, the system extracts embeddings, computes cosine similarity, and predicts same/different at the threshold that maximizes accuracy.

Two scenarios tested:
- **Both degraded:** worst case where both family and hospital photos are degraded
- **One degraded:** realistic case where family has clean reference photo, hospital photo is degraded

**Metrics reported:**
- Accuracy on successfully processed pairs
- Effective accuracy (counting face detection failures as wrong predictions)
- AUC of ROC curve
- Face detection rate
- Best threshold

## Results

### Clean baseline

| Metric | Value |
|--------|-------|
| Accuracy | 99.60% |
| AUC | 0.9979 |
| Best threshold | 0.206 |
| Skipped pairs | 0 |

ArcFace pretrained baseline matches published LFW results, confirming the pipeline is correctly implemented.

### Degraded baseline — both images degraded

| Severity | Accuracy | Skipped | Effective Accuracy | Detection Rate |
|----------|----------|---------|-------------------|----------------|
| Light    | 98.92%   | 74      | 91.6%             | 92.6%          |
| Medium   | 98.16%   | 76      | 90.7%             | 92.4%          |
| Heavy    | 91.64%   | 689     | **28.5%**         | **31.1%**      |

### Degraded baseline — one image degraded (realistic scenario)

| Severity | Accuracy | Skipped | Effective Accuracy | Detection Rate |
|----------|----------|---------|-------------------|----------------|
| Light    | 99.37%   | 46      | 94.8%             | 95.4%          |
| Medium   | 99.48%   | 37      | 95.8%             | 96.3%          |
| Heavy    | 96.83%   | 464     | **51.9%**         | **53.6%**      |

## Key Findings

### 1. Face matching is robust on what it can see

When ArcFace successfully detects a face, matching accuracy remains high even under heavy degradation (91.6% on heavy_both, 96.8% on heavy_one). The matching component itself is not the failure point.

### 2. Face detection is the critical failure mode

Under heavy degradation with both images degraded, face detection fails on **68.9%** of images. This is the dominant cause of system failure, not poor matching quality. Under realistic conditions (one image degraded), detection still fails on 46.4% of images.

### 3. The 71 percentage point gap

Clean baseline effective accuracy: 99.6%  
Heavy degradation effective accuracy: 28.5%  
**Drop: 71.1 percentage points**

This gap quantifies the problem this project addresses.

### 4. Implication for system design

Fine-tuning the face matching model would offer marginal improvement, since matching is already strong on detected faces. The fundamental issue is that no face matching can occur when face detection fails. This motivates a multimodal approach where alternative identifiers (body markers, physical descriptors) provide complementary signals when face detection is unavailable.

## Decisions Made

Based on these findings:

1. **No face fine-tuning in Month 1.** The face matching component is performing as well as can be expected; fine-tuning would address the wrong problem.

2. **Project scope refined.** From "improve face recognition for mass casualty events" to "build a multimodal identification system that works when face recognition cannot."

3. **Marker categories expanded.** Beyond tattoos to include piercings, birthmarks, scars, moles, and other distinguishing physical features, recognizing demographic and cultural variation in marker availability.

4. **CASIA-WebFace deferred to Month 3.** Final fine-tuning experiments (if any) will use a larger dataset in Month 3 once the full multimodal pipeline is validated.

## Artifacts Produced

- `src/data/datasets.py` — LFW dataset classes
- `src/data/download.py` — automated dataset download
- `src/data/augmentations.py` — mass casualty degradation pipeline
- `src/features/face_extractor.py` — ArcFace embedding extraction
- `src/utils/metrics.py` — verification metrics with effective accuracy
- `experiments/01_face_baseline.py` — clean baseline evaluation
- `experiments/02_face_degraded_baseline.py` — degraded baseline evaluation
- `experiments/results/01_face_baseline.json` — clean baseline results
- `experiments/results/02_face_degraded_baseline.json` — degraded baseline results
- `experiments/results/degradation_examples.png` — visualization of degradations

## Limitations Noted

1. **Synthetic degradations are an approximation.** While our augmentations are informed by documented mass casualty conditions, they cannot fully replicate the variability of real victim photographs.

2. **LFW demographic imbalance.** LFW is known to be biased toward Western, light-skinned, male subjects. Generalization to underrepresented groups requires further validation.

3. **Single dataset evaluation.** All results are on LFW pairs. Validation on additional datasets (IJB-C, CASIA, real disaster image collections where ethically available) is future work.

4. **No real-world deployment validation.** Results are research findings on synthetic data and should not be interpreted as deployment-ready performance metrics.

## Next Steps — Month 2

The face baseline confirms the need for multimodal identification. Month 2 focuses on building the body marker detection component:

- Request NIST Tatt-C dataset access
- Explore dermatology datasets for skin marker detection
- Plan annotation strategy for piercings and scars
- Begin fine-tuning YOLOv8 for multi-marker detection
- Build marker embedding model for matching

The same rigor applied in Month 1 will be applied to Month 2: establish baseline, validate datasets, document decisions, produce reproducible artifacts.

---

*Document compiled at end of Month 1. All results reproducible via scripts in `experiments/`.*