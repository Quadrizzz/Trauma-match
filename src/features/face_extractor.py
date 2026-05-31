"""
Face embedding extraction using ArcFace via InsightFace.
"""
import numpy as np
import cv2
from insightface.app import FaceAnalysis


class FaceExtractor:
    """Extracts face embeddings using a pre-trained ArcFace model."""
    
    def __init__(self, model_name="buffalo_l", providers=None, det_size=(160, 160), det_thresh=0.3):
        providers = providers or ['CPUExecutionProvider']
        self.app = FaceAnalysis(
            name=model_name, 
            providers=providers,
            allowed_modules=['detection', 'recognition'],  # skip landmarks/age — faster
        )
        ctx_id = 0 if 'CUDAExecutionProvider' in providers else -1
        self.app.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=det_thresh)
        
    def extract(self, image):
        """
        Extract face embedding from an image.
        
        Args:
            image: numpy array of shape (H, W, 3), RGB or BGR
        
        Returns:
            embedding: numpy array of shape (512,) or None if no face detected
        """
        # InsightFace expects BGR (OpenCV format)
        if image.shape[-1] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        faces = self.app.get(image_bgr)
        
        if len(faces) == 0:
            return None
        
        # if multiple faces detected, take the largest (highest detection score)
        face = max(faces, key=lambda f: f.det_score)
        return face.normed_embedding  # already L2-normalized
    
    def similarity(self, embedding1, embedding2):
        """Cosine similarity between two L2-normalized embeddings."""
        return float(np.dot(embedding1, embedding2))


if __name__ == "__main__":
    # quick sanity check
    from src.data.datasets import LFWPairsDataset
    
    print("Loading face extractor...")
    extractor = FaceExtractor()
    
    print("Loading first pair from LFW...")
    pairs = LFWPairsDataset()
    img1, img2, is_same = pairs[5]
    
    print("Extracting embeddings...")
    emb1 = extractor.extract(img1)
    emb2 = extractor.extract(img2)
    
    if emb1 is None or emb2 is None:
        print("Could not detect face in one of the images.")
    else:
        print(f"Embedding shape: {emb1.shape}")
        sim = extractor.similarity(emb1, emb2)
        print(f"Similarity: {sim:.4f}")
        print(f"Ground truth same person: {bool(is_same)}")