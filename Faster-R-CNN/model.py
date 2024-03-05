import os
import torch
import logging
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score

from ChestXrayDataset import ChestXrayDataset  # Ensure this script is properly defined

# Setup logging for monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ChestXRayTraining')

# Configuration settings
config = {
    'hdf5_file': '/mnt/beegfs/home/s425500/chest-x-ray-abnormalities-detection/dicom_images_final.hdf5',
    'train_csv': '/mnt/beegfs/home/s425500/chest-x-ray-abnormalities-detection/data/train.csv',
    'test_csv': '/mnt/beegfs/home/s425500/chest-x-ray-abnormalities-detection/data/test.csv',
    'num_classes': 15,  # Including background
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'batch_size': 4,
    'num_epochs': 10,
    'learning_rate': 0.005,
    'checkpoint_dir': './model_checkpoints',  # Ensure this directory exists
}

def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Handle variable-sized bounding boxes or targets here
    # For example, padding the bounding boxes to match the largest tensor size in the batch

    images = torch.stack(images, 0)
    # Ensure targets are appropriately handled as well

    return images, targets


# Data preparation
train_dataset = ChestXrayDataset(config['train_csv'], config['hdf5_file'], target_size=(224, 224), phase='train')
test_dataset = ChestXrayDataset(config['test_csv'], config['hdf5_file'], target_size=(224, 224), phase='test')

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)

# Model setup
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, config['num_classes'])
model.to(config['device'])

# Optimizer and scheduler
optimizer = SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9, weight_decay=0.005)
lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
scaler = GradScaler()  # For AMP

# Function to save model checkpoints
def save_model(model, path):
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")

# Evaluation function
def evaluate_model(model, data_loader, device):
    model.eval()
    all_detections = []
    all_ground_truths = []
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = list(img.to(device) for img in images)
            outputs = model(images)
            
            for i, output in enumerate(outputs):
                # Assuming outputs are already moved to CPU
                scores = output['scores'].detach().cpu().numpy()
                pred_labels = output['labels'].detach().cpu().numpy()
                true_labels = targets[i]['labels'].detach().cpu().numpy()
                
                # Apply a score threshold to filter out low-confidence detections
                threshold = 0.5
                valid_indices = scores > threshold
                valid_scores = scores[valid_indices]
                valid_pred_labels = pred_labels[valid_indices]
                
                # Append detections and ground truths for each image
                all_detections.extend(valid_pred_labels)
                all_ground_truths.extend(true_labels)
    
    # Handle edge case where no detections are made
    if not all_detections:
        print("No detections were made.")
        return 0
    
    # Calculate the average precision score
    average_precision = average_precision_score(all_ground_truths, all_detections, average='macro')
    
    model.train()
    return average_precision

# Training loop with validation and checkpoint saving
best_val_mAP = 0.0
for epoch in range(config['num_epochs']):
    model.train()
    for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"):
        images = [img.to(config['device']) for img in images]
        targets_converted = []
        for t in targets:
            t_converted = {k: torch.tensor(v, device=config['device']) if isinstance(v, np.ndarray) else v.to(config['device']) for k, v in t.items()}
            targets_converted.append(t_converted)
        targets = targets_converted

        optimizer.zero_grad()
        with autocast():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

    lr_scheduler.step()

    # Evaluate and save model conditionally
    if epoch % 2 == 0:
        val_mAP = evaluate_model(model, test_loader, config['device'])
        logger.info(f"Epoch {epoch+1}, Validation mAP: {val_mAP:.4f}")
        if val_mAP > best_val_mAP:
            best_val_mAP = val_mAP
            save_model_path = os.path.join(config['checkpoint_dir'], f"best_model_epoch_{epoch+1}.pth")
            save_model(model, save_model_path)

logger.info("Training completed.")



