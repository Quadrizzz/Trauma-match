"""
End-to-end marker matching: detection → crop → embedding → similarity.
"""
import numpy as np


def extract_marker_embeddings(image, detector, extractor, min_box_size=20):
    """
    Detect markers in an image and extract a CLIP embedding for each.
    
    Args:
        image: numpy array (H, W, 3) RGB
        detector: MarkerDetector instance
        extractor: TattooExtractor instance
        min_box_size: minimum pixel dimension for a valid detection
    
    Returns:
        list of dicts with 'embedding', 'box', 'label', 'score', 'queried_type'
    """
    detections = detector.detect(image)
    
    marker_data = []
    for det in detections:
        x1, y1, x2, y2 = det['box']
        if (x2 - x1) < min_box_size or (y2 - y1) < min_box_size:
            continue
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        embedding = extractor.extract(crop)
        marker_data.append({
            'embedding': embedding,
            'box': det['box'],
            'label': det['label'],
            'score': det['score'],
            'queried_type': det['queried_type'],
        })
    return marker_data


def marker_similarity(markers_a, markers_b, extractor):
    """
    Maximum cosine similarity between any pair of markers from two images.
    
    Returns None if either side has no markers.
    """
    if not markers_a or not markers_b:
        return None
    
    max_sim = -1.0
    for m_a in markers_a:
        for m_b in markers_b:
            sim = extractor.similarity(m_a['embedding'], m_b['embedding'])
            if sim > max_sim:
                max_sim = sim
    return max_sim