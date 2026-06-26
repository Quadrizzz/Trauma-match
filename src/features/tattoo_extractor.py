"""
Tattoo/marker embedding extraction using frozen CLIP.
"""
import torch
import open_clip
import numpy as np
from PIL import Image


class TattooExtractor:
    """Extracts marker embeddings using a pretrained CLIP image encoder."""
    
    def __init__(self, model_name='ViT-B-32', pretrained='laion2b_s34b_b79k', device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device).eval()
    
    @torch.no_grad()
    def extract(self, image):
        """Extract L2-normalized embedding from an image."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        embedding = self.model.encode_image(img_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().squeeze()
    
    def similarity(self, embedding1, embedding2):
        """Cosine similarity between two L2-normalized embeddings."""
        return float(np.dot(embedding1, embedding2))