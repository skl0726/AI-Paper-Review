import torch
import torch.nn as nn
import torch.nn.functional as F
from anchor import make_center_anchors
from utils import intersection_over_union, center_to_corner, corner_to_center

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class YOLOv2Loss(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = [(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)]

    def make_target(self, gt_boxes, gt_labels, pred_xy, pred_wh):
        out_size = pred_xy.size(2)
        batch_size = pred_xy.size(0)
        resp_mask = torch.zeros([batch_size, out_size, out_size, 5])
        gt_xy = torch.zeros([batch_size, out_size, out_size, 5, 2])
        gt_wh = torch.zeros([batch_size, out_size, out_size, 5, 2])
        gt_conf = torch.zeros([batch_size, out_size, out_size, 5])
        gt_cls = torch.zeros([batch_size, out_size, out_size, 5, self.num_classes])

        center_anchors = make_center_anchors(anchors_wh=self.anchors, grid_size=out_size)
        corner_anchors = center_to_corner(center_anchors).view(out_size * out_size * 5, 4)

        for b in range(batch_size):
            label = gt_labels[b]
            corner_gt_box = gt_boxes[b]
            corner_gt_box_13 = corner_gt_box * float(out_size)

            center_gt_box = corner_to_center(corner_gt_box)
            center_gt_box_13 = center_gt_box * float(out_size)

            bxby = center_gt_box_13[..., :2] # [# obj, 2]
            x_y_ = bxby - bxby.floor()       # [# obj, 2] 0~1 scale
            bwbh = center_gt_box_13[..., 2:]

            iou_anchors_gt = intersection_over_union(corner_anchors, corner_gt_box_13) # [845, # obj]
            iou_anchors_gt = iou_anchors_gt.view(out_size, out_size, 5, -1)

            num_obj = corner_gt_box.size(0)

            for n_obj in range(num_obj):
                cx, cy = bxby[n_obj]
                cx = int(cx)
                cy = int(cy)

                _, max_idx = iou_anchors_gt[cx, cy, :, n_obj].max(0) # which anchor as maximum iou?
                j = max_idx
                resp_mask[b, cx, cy, j] = 1
                gt_xy[b, cx, cy, j, :] = x_y_[n_obj]
                w_h_ = bwbh[n_obj] / torch.FloatTensor(self.anchors[j].to(device)) # ratio
                gt_wh[b, cx, cy, j, :] = w_h_
                gt_cls[b, cx, cy, j, int(label[n_obj].item())] = 1

            pred_xy_ = pred_xy[b]
            pred_wh_ = pred_wh[b]
            center_pred_xy = center_anchors[..., :2].floor() + pred_xy_       # [845, 2]
            center_pred_wh = center_anchors[..., 2:] * pred_wh_               # [845, 2]
            center_pred_bbox = torch.cat([center_pred_xy, center_pred_wh], dim=-1)
            corner_pred_bbox = center_to_corner(center_pred_bbox).view(-1, 4) # [845, 4]

            iou_pred_gt = intersection_over_union(corner_pred_bbox, corner_gt_box_13) # [845, # obj]
            iou_pred_gt = iou_pred_gt.view(out_size, out_size, 5, -1)

            gt_conf[b] = iou_pred_gt.max(-1)[0] # each obj, maximum preds [13, 13, 5]

        return resp_mask, gt_xy, gt_wh, gt_conf, gt_cls

    def forward(self, pred_targets, gt_boxes, gt_labels):
        """
        pred_targets: [B, 13, 13, 125]
        gt_boxes: [B, 4]
        """

        out_size = pred_targets.size(1)
        pred_targets = pred_targets.view(-1, out_size, out_size, 5, 5 + self.num_classes)
        pred_xy = pred_targets[..., :2].sigmoid() # sigmoid(tx ty)
        pred_wh = pred_targets[..., 2:4].exp()
        pred_conf = pred_targets[..., 4].sigmoid()
        pred_cls = pred_targets[..., 5:]

        resp_mask, gt_xy, gt_wh, gt_conf, gt_cls = self.make_target(gt_boxes, gt_labels, pred_xy, pred_wh)

        xy_loss = resp_mask.unsqueeze(-1).expand_as(gt_xy) * (gt_xy - pred_xy.cpu()) ** 2

        wh_loss = resp_mask.unsqueeze(-1).expand_ax(gt_wh) * (torch.sqrt(gt.wh) - torch.sqrt(pred_wh.cpu())) ** 2

        conf_loss = resp_mask * (gt_conf - pred_conf.cpu()) ** 2

        no_conf_loss = (1 - resp_mask) * (gt_conf - pred_conf.cpu()) ** 2

        pred_cls = F.softmax(pred_cls, dim=-1) # [N * 13 * 13 * 5, 20]
        resp_cell = resp_mask.max(-1)[0].unsqueeze(-1).unsqueeze(-1).expand_as(gt_cls) # [B, 13, 13, 5, 20]
        cls_loss = resp_cell * (gt_cls - pred_cls.cpu()) ** 2

        loss1 = 5 * xy_loss.sum()
        loss2 = 5 * wh_loss.sum()
        loss3 = 1 * conf_loss.sum()
        loss4 = 0.5 * no_conf_loss.sum()
        loss5 = 1 * cls_loss.sum()

        return loss1 + loss2 + loss3 + loss4 + loss5, (loss1, loss2, loss3, loss4, loss5)

"""
if __name__ == '__main__':
    image = torch.randn([5, 3, 416, 416])
    pred = torch.zeros([5, 13, 13, 125]) # batch, 13, 13, etc.
    criterion = YOLOv2Loss(num_classes=20)
"""
