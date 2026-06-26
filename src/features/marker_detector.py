"""
Multi-class marker detection using Grounding DINO.
Detects tattoos, piercings, scars, birthmarks, and moles.
"""
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


MARKER_TYPES = ["tattoo", "piercing", "scar", "birthmark", "mole"]


def _box_iou(box1, box2):
    """Compute IoU between two boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    inter_x1, inter_y1 = max(x1_1, x1_2), max(y1_1, y1_2)
    inter_x2, inter_y2 = min(x2_1, x2_2), min(y2_1, y2_2)
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


def _deduplicate(detections, iou_threshold=0.5):
    """Remove overlapping detections, keeping highest confidence."""
    if not detections:
        return []
    sorted_dets = sorted(detections, key=lambda d: d['score'], reverse=True)
    keep = []
    for det in sorted_dets:
        if any(_box_iou(det['box'], k['box']) > iou_threshold for k in keep):
            continue
        keep.append(det)
    return keep


class MarkerDetector:
    """
    Zero-shot marker detection using Grounding DINO with per-class queries.
    
    Per-class querying significantly outperforms multi-class prompting because
    competing text tokens reduce per-class detection scores below useful thresholds.
    """
    
    def __init__(self, model_id="IDEA-Research/grounding-dino-tiny", 
                 box_threshold=0.50, text_threshold=0.50, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def _detect_single(self, image, prompt):
        """Run detection with a single-class prompt."""
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[pil_image.size[::-1]],
        )[0]
        
        detections = []
        marker_type = prompt.rstrip('.').strip()
        for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            detections.append({
                'box': (int(x1), int(y1), int(x2), int(y2)),
                'label': label if label else marker_type,
                'score': float(score.cpu()),
                'queried_type': marker_type,
            })
        return detections
    
    def detect(self, image, marker_types=None):
        """
        Detect all marker types in an image using per-class queries.
        
        Args:
            image: numpy array (H, W, 3) RGB
            marker_types: list of marker types to detect. Defaults to all.
        
        Returns:
            list of detection dicts (deduplicated by IoU)
        """
        marker_types = marker_types or MARKER_TYPES
        all_detections = []
        for marker_type in marker_types:
            detections = self._detect_single(image, f"{marker_type}.")
            all_detections.extend(detections)
        return _deduplicate(all_detections)