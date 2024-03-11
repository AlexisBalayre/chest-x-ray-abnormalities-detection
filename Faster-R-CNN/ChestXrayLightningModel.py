import lightning as L
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
import torch.nn.functional as F
import torchmetrics


class ChestXrayLightningModel(L.LightningModule):
    def __init__(self, num_classes, learning_rate=0.05):
        super().__init__()

        self.model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        self.learning_rate = learning_rate

        self.val_metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True, iou_thresholds=0.4)

        self.cosine_t_max = 10
        self.save_hyperparameters(ignore=["model"])

    def forward(self, inputs):
        return self.model(inputs)
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        self.log_dict(
            loss_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        train_loss = sum(loss for loss in loss_dict.values())
        self.log(
            "train_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return train_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        pred = self.model(images)
        self.val_metric.update(preds=pred, target=targets)
    
    def on_validation_epoch_end(self):
        mAPs = self.val_metric.compute()
        map_per_class = mAPs.pop("map_per_class")
        mar_100_per_class = mAPs.pop("mar_100_per_class")
        classes = mAPs.pop("classes")
        map = mAPs.pop("map")
        self.log("val_map", map, on_step=False, on_epoch=True, prog_bar=True, logger=True)
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
        optimizer = torch.optim.SGD(self.model.parameters(),
            lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cosine_t_max)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "step", # step means "batch" here, default: epoch   # New!
                "frequency": 1, # default
            },
        }


        


