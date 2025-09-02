import time
from tqdm.notebook import tqdm
import torch
from PIL import Image
from ultralytics import YOLO

# Load the baseline YOLOv8n model
model_v8n = YOLO("/content/yolov8n.pt")

# Evaluate the baseline model
metrics_v8n = model_v8n.val(device=0, data=’coco128.yaml’)
print(f"Baseline YOLOv8n Metrics:\n{metrics_v8n}")

# Load the proposed enhanced YOLOv8 model
model_proposed = YOLO("/content/proposed_model.pt") # Replace with your model’s path

# Evaluate the proposed model
metrics_proposed = model_proposed.val(device=0, data=proposed_dataset.yaml’)
print(f"Proposed Model Metrics:\n{metrics_proposed}")


"""## Pruning"""

def prune_model(model, amount=0.3):
    """Prunes the given model using L1 unstructured pruning."""
    import torch.nn.utils.prune as prune
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name=’weight’, amount=amount)
            prune.remove(m, “weight”)

# Prune the baseline YOLOv8n model
prune_model(model_v8n.model.model, 0.3)

# Evaluate the pruned baseline model
metrics_v8n_pruned = model_v8n.val(device=0, data=’coco128.yaml’)
print(f"Pruned YOLOv8n Metrics:\n{metrics_v8n_pruned}")

# Prune the proposed model
prune_model(model_proposed.model.model, 0.3)  # Assumes similar internal structure

# Evaluate the pruned proposed model
metrics_proposed_pruned = model_proposed.val(device=0, data=proposed_dataset.yaml’)
print(f"Pruned Proposed Model Metrics:\n{metrics_proposed_pruned}")
