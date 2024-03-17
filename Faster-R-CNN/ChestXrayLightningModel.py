import lightning as L
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
import torch.nn.functional as F
import torchmetrics


class ChestXrayLightningModel(L.LightningModule):
    """
    A LightningModule for chest X-ray detection using a Faster R-CNN model with a ResNet50 backbone.

    This module is designed to be used for the detection of abnormalities in chest X-ray images,
    utilizing a pre-trained Faster R-CNN model with custom modifications for the task-specific
    number of classes.

    Parameters:
    - num_classes (int): Number of classes for detection, including the background class.
    - learning_rate (float, optional): Initial learning rate for the optimizer.
    - cosine_t_max (int, optional): Maximum number of iterations for the cosine annealing scheduler.

    Attributes:
    - model (torch.nn.Module): The Faster R-CNN model with a ResNet50 backbone.
    - learning_rate (float): Learning rate for the optimizer.
    - cosine_t_max (int): Maximum number of iterations for the cosine annealing scheduler.
    - val_metric (MeanAveragePrecision): Metric for validation, calculating mean average precision.
    """

    def __init__(self, num_classes, learning_rate=0.01, cosine_t_max=20):
        super().__init__()

        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        )
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        self.learning_rate = learning_rate
        self.cosine_t_max = cosine_t_max

        self.val_metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

        self.save_hyperparameters(ignore=["model"])

    def forward(self, inputs):
        """
        Forward pass of the model.

        Parameters:
        - inputs (list of torch.Tensor): List of images to perform detection on.

        Returns:
        - dict: The model's predictions including detected boxes, labels, and scores.
        """
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        """
        Defines the training logic for a single batch of data.

        Parameters:
        - batch (tuple): The batch to train on, containing images and their respective targets.
        - batch_idx (int): The index of the current batch.

        Returns:
        - torch.Tensor: The aggregated loss from the Faster R-CNN model.
        """
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        self.log_dict(
            loss_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        train_loss = sum(loss for loss in loss_dict.values())
        self.log(
            "train_loss",
            train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return train_loss

    def validation_step(self, batch, batch_idx):
        """
        Defines the validation logic for a single batch of data.

        Parameters:
        - batch (tuple): The batch to validate on, containing images and their respective targets.
        - batch_idx (int): The index of the current batch.
        """
        images, targets = batch
        pred = self.model(images)
        self.val_metric.update(preds=pred, target=targets)

    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch to log the mean average precision (mAP) and
        mean average recall (mAR) metrics.
        """
        mAPs = self.val_metric.compute()
        map_per_class = mAPs.pop("map_per_class")
        mar_100_per_class = mAPs.pop("mar_100_per_class")
        classes = mAPs.pop("classes")
        map = mAPs.pop("map")
        self.log(
            "val_map", map, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_dict(mAPs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        try:
            for i, class_name in enumerate(classes):
                self.log(
                    f"mAP_{class_name}",
                    map_per_class[i],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )
                self.log(
                    f"mar_100_{class_name}",
                    mar_100_per_class[i],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )
        except:
            pass
        self.val_metric.reset()

    def configure_optimizers(self):
        """
        Sets up the optimizer and learning rate scheduler to be used during training.

        Returns:
        - dict: A dictionary containing the optimizer and LR scheduler configurations.
        """
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=0.0005,
        )  # SGD optimizer with momentum and weight decay
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cosine_t_max, eta_min=0.0001
        )  # Cosine annealing learning rate scheduler

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
