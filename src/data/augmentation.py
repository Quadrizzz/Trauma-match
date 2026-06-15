import albumentations as A
import numpy as np
import cv2


def add_medical_occlusion(image, region='lower_face'):
    """Add horizontal medical occlusion to a specific facial region."""
    img = image.copy()
    h, w = img.shape[:2]
    
    color = (
        int(np.random.randint(200, 230)),
        int(np.random.randint(195, 225)),
        int(np.random.randint(190, 220)),
    )
    
    if region == 'lower_face':
        y1 = int(h * 0.52); y2 = int(h * 0.78)
        x1 = int(w * 0.15); x2 = int(w * 0.85)
    elif region == 'forehead':
        y1 = int(h * 0.08); y2 = int(h * 0.28)
        x1 = int(w * 0.10); x2 = int(w * 0.90)
    elif region == 'cheek':
        side = np.random.choice(['left', 'right'])
        y1 = int(h * 0.42); y2 = int(h * 0.62)
        if side == 'left':
            x1 = int(w * 0.05); x2 = int(w * 0.35)
        else:
            x1 = int(w * 0.65); x2 = int(w * 0.95)
    elif region == 'chin':
        y1 = int(h * 0.78); y2 = int(h * 0.98)
        x1 = int(w * 0.15); x2 = int(w * 0.85)
    
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    return img


def add_blood_stain(image, num_stains=None):
    """Add irregular dark red patches to simulate blood/bruising."""
    img = image.copy()
    h, w = img.shape[:2]
    
    if num_stains is None:
        num_stains = np.random.randint(1, 4)
    
    for _ in range(num_stains):
        color = (
            int(np.random.randint(20, 60)),
            int(np.random.randint(20, 50)),
            int(np.random.randint(80, 140)),
        )
        cx = np.random.randint(int(w * 0.2), int(w * 0.8))
        cy = np.random.randint(int(h * 0.3), int(h * 0.8))
        rx = np.random.randint(8, 25)
        ry = np.random.randint(8, 22)
        cv2.ellipse(img, (cx, cy), (rx, ry), 
                    np.random.randint(0, 180), 0, 360, color, -1)
    return img


def add_dirt_layer(image, intensity=0.4):
    """Add a dust/dirt layer across the face (collapse, debris scenarios)."""
    img = image.copy()
    h, w = img.shape[:2]
    
    # generate noise-based dirt texture
    dirt = np.random.normal(loc=0, scale=30, size=(h, w, 3))
    
    # warm brown/grey tint for dust
    tint = np.array([
        np.random.randint(80, 130),    # B
        np.random.randint(100, 150),   # G  
        np.random.randint(120, 170),   # R (warm dust color)
    ])
    
    # combine dirt texture with tint
    dirt_layer = np.clip(dirt + tint, 0, 255).astype(np.uint8)
    
    # blend with original image
    img = cv2.addWeighted(img, 1 - intensity, dirt_layer, intensity, 0)
    return img


def add_swelling_distortion(image, severity=0.5):
    """Simulate facial swelling using elastic warping."""
    img = image.copy()
    h, w = img.shape[:2]
    
    # alpha controls displacement magnitude, sigma controls smoothness
    alpha = int(20 * severity)
    sigma = 5
    
    # use albumentations elastic transform
    transform = A.ElasticTransform(alpha=alpha, sigma=sigma, p=1.0)
    img = transform(image=img)['image']
    return img


def add_wet_reflection(image, intensity=0.3):
    """Add bright reflective patches simulating sweat/blood on skin."""
    img = image.copy()
    h, w = img.shape[:2]
    
    num_reflections = np.random.randint(1, 3)
    for _ in range(num_reflections):
        cx = np.random.randint(int(w * 0.3), int(w * 0.7))
        cy = np.random.randint(int(h * 0.3), int(h * 0.7))
        radius = np.random.randint(10, 25)
        
        # bright spot
        overlay = img.copy()
        cv2.circle(overlay, (cx, cy), radius, (240, 240, 240), -1)
        img = cv2.addWeighted(img, 1 - intensity, overlay, intensity, 0)
    return img


def _compatible_regions(used_regions):
    """Return regions that don't conflict with already-used ones."""
    conflicts = {
        'lower_face': ['chin'],
        'chin': ['lower_face'],
        'forehead': [],
        'cheek': [],
    }
    all_regions = ['lower_face', 'forehead', 'cheek', 'chin']
    available = []
    for r in all_regions:
        if r in used_regions:
            continue
        if any(used in conflicts.get(r, []) for used in used_regions):
            continue
        available.append(r)
    return available


def apply_medical_degradation(image, severity='medium'):
    """
    Apply realistic mass casualty degradations.
    
    Severity levels reflect different real-world scenarios:
    - light:  minor injury, well-lit conditions (clinic, hospital lobby)
    - medium: moderate trauma, suboptimal conditions (ambulance, ER)
    - heavy:  severe trauma, chaotic conditions (disaster scene)
    """
    img = image.copy()
    
    config = {
        'light': {
            'brightness_limit': (-0.15, -0.05),
            'blur_limit': 5,
            'blur_prob': 0.5,
            'occlusion_prob': 0.5,
            'num_occlusions': 1,
            'blood_prob': 0.4,
            'dirt_prob': 0.2,
            'swelling_prob': 0.0,
            'wet_prob': 0.4,
        },
        'medium': {
            'brightness_limit': (-0.25, -0.05),
            'blur_limit': 9,
            'blur_prob': 0.7,
            'occlusion_prob': 0.75,
            'num_occlusions': 1,
            'blood_prob': 0.6,
            'dirt_prob': 0.4,
            'swelling_prob': 0.3,
            'wet_prob': 0.4,
        },
        'heavy': {
            'brightness_limit': (-0.35, -0.15),
            'blur_limit': 15,
            'blur_prob': 0.9,
            'occlusion_prob': 0.95,
            'num_occlusions': 2,
            'blood_prob': 0.8,
            'dirt_prob': 0.6,
            'swelling_prob': 0.6,
            'wet_prob': 0.6,
        },
    }[severity]
    
    # base: blur and lighting (rushed photography in poor conditions)
    pipeline = A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=config['brightness_limit'],
            contrast_limit=(-0.15, 0),
            p=0.6,
        ),
        A.OneOf([
            A.MotionBlur(blur_limit=config['blur_limit'], p=1.0),
            A.GaussianBlur(blur_limit=(3, config['blur_limit']), p=1.0),
        ], p=config['blur_prob']),
    ])
    img = pipeline(image=img)['image']
    
    # swelling (structural distortion from trauma)
    if np.random.random() < config['swelling_prob']:
        img = add_swelling_distortion(img, severity=0.5 if severity == 'medium' else 0.8)
    
    # blood/bruising
    if np.random.random() < config['blood_prob']:
        img = add_blood_stain(img)
    
    # dirt/dust (debris, collapse scenarios)
    if np.random.random() < config['dirt_prob']:
        img = add_dirt_layer(img, intensity=0.3 if severity == 'medium' else 0.45)
    
    # wet reflection (sweat, blood on skin)
    if np.random.random() < config['wet_prob']:
        img = add_wet_reflection(img, intensity=0.25 if severity == 'medium' else 0.35)
    
    # medical occlusions (applied last so they cover everything else)
    if np.random.random() < config['occlusion_prob']:
        used_regions = []
        for _ in range(config['num_occlusions']):
            available = _compatible_regions(used_regions)
            if not available:
                break
            region = np.random.choice(available)
            img = add_medical_occlusion(img, region=region)
            used_regions.append(region)
    
    return img


def get_degradation_pipeline(severity='medium'):
    def transform(image):
        return {'image': apply_medical_degradation(image, severity)}
    return transform


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from src.data.datasets import LFWPairsDataset
    
    pairs = LFWPairsDataset()
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    
    for row in range(4):
        img, _, _ = pairs[row * 30]
        axes[row, 0].imshow(img)
        axes[row, 0].set_title("Original" if row == 0 else "")
        axes[row, 0].axis('off')
        
        for col, severity in enumerate(['light', 'medium', 'heavy']):
            degraded = apply_medical_degradation(img, severity)
            axes[row, col + 1].imshow(degraded)
            axes[row, col + 1].set_title(severity.capitalize() if row == 0 else "")
            axes[row, col + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('experiments/results/degradation_examples.png', dpi=100)
    plt.show()