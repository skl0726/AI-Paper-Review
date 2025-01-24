import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")

        self.S = S # S is split size of image
        self.B = B # B is number of boxes
        self.C = C # C is number of classes

        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # Predictions are shaped (BATCH_SIZE, S*S*(C+B*5)) when inputted
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5)

        # Calculate IoU for the two predicted bounding boxes with target box
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25]) # predictions[..., 21:25]: coordinate value of first bbox
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25]) # predictions[..., 26:30]: coordinate value of second bbox
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # Take the box with highest IoU out of the two prediction
        # Note that bestbox will be indices of 0, 1 for which bbox was best
        iou_maxes, bestbox = torch.max(ious, dim=0) # bestbox -> index of box which have larger value of IoU
        exists_box = target[..., 20].unsqueeze(3) # in paper, this is 1obj_ij

        """ FOR BOX COORDINATES """
        # Set boxes with no object in them to 0. We only take out one of the two predictions,
        # which is the one with highest IoU calculated previously.
        box_predictions = exists_box * (bestbox * predictions[..., 26:30] + (1 - bestbox) * predictions[..., 21:25])
        box_targets = exists_box * target[..., 21:25]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        """ FOR OBJECT LOSS """
        # pred_box is the confidence score for the bbox with highest IoU
        pred_box = (bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21])
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )

        """ FOR NO OBJECT LOSS """
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        """ FOR CLASS LOSS """
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2),
        )

        loss = (
            self.lambda_coord * box_loss # first two rows in paper
            + object_loss # third row in paper
            + self.lambda_noobj * no_object_loss # fourth row
            + class_loss # fifth row
        )

        return loss
