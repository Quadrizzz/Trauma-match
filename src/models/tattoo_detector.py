"""
YOLOv8 tattoo detector training and inference.
"""
from pathlib import Path
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TATTOO_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "tattoo_detection"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints" / "tattoo"


def train_detector(
    data_yaml_path,
    base_model="yolov8n.pt",
    epochs=100,
    imgsz=640,
    batch=16,
    device=None,
):
    """
    Train YOLOv8 on tattoo detection dataset.
    
    Args:
        data_yaml_path: path to data.yaml from Roboflow export
        base_model: pretrained model to fine-tune (yolov8n.pt is fastest)
        epochs: number of training epochs
        imgsz: input image size
        batch: batch size
        device: 'cuda', 'cpu', or None (auto-detect)
    """
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    
    model = YOLO(base_model)
    
    results = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(CHECKPOINTS_DIR),
        name="tattoo_yolov8",
        exist_ok=True,
        patience=20,  # early stopping
        save=True,
        plots=True,
    )
    
    return results


def evaluate_detector(model_path, data_yaml_path):
    """Evaluate trained detector on test set."""
    model = YOLO(model_path)
    metrics = model.val(data=str(data_yaml_path), split='test')
    
    print(f"\n=== TATTOO DETECTION RESULTS ===")
    print(f"  mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  Precision:    {metrics.box.mp:.4f}")
    print(f"  Recall:       {metrics.box.mr:.4f}")
    
    return metrics


class TattooDetector:
    """Wrapper for using the trained YOLO detector at inference time."""
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def detect(self, image, conf=0.25):
        """
        Detect tattoos in an image.
        
        Args:
            image: numpy array (H, W, 3) RGB
            conf: confidence threshold
        
        Returns:
            list of bounding boxes [(x1, y1, x2, y2, confidence), ...]
        """
        results = self.model(image, conf=conf, verbose=False)
        boxes = []
        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf_val = float(box.conf[0])
                boxes.append((
                    int(xyxy[0]), int(xyxy[1]),
                    int(xyxy[2]), int(xyxy[3]),
                    conf_val,
                ))
        return boxes
    
    def crop_tattoo(self, image, box):
        """Extract the tattoo region from an image given a bounding box."""
        x1, y1, x2, y2 = box[:4]
        return image[y1:y2, x1:x2]