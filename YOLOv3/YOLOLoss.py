import torch
import torch.nn as nn

class YOLOLoss(nn.Module):
    def __init__(self, anchor_boxes, n_classes, iou_threshold=0.4):
        super(YOLOLoss, self).__init__()

        self.anchor_boxes = anchor_boxes
        self.n_classes = n_classes
        self.iou_threshold = iou_threshold
        self.bceloss = nn.BCELoss()

    def __refine_output__(self, pred, anchors, classes):
        # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
        grid_size = pred.shape[1:3]
        box_xy, box_wh, objectness, class_probs = torch.split(pred, (2, 2, 1, classes), dim=-1)

        box_xy = box_xy.detach()
        box_wh = box_wh.detach()
        objectness = objectness.detach()
        class_probs = class_probs.detach()

        box_xy = torch.sigmoid_(box_xy)
        objectness = torch.sigmoid_(objectness)
        class_probs = torch.sigmoid_(class_probs)
        pred_box = torch.cat((box_xy, box_wh), axis=-1)  # original xywh for loss

        grid = torch.meshgrid(torch.arange(0, grid_size[1]), torch.arange(0, grid_size[0]))
        grid = torch.unsqueeze(torch.stack(grid, axis=-1), axis=2).cuda()  # shape: [gx, gy, 1, 2]

        box_xy = (box_xy + grid) / torch.tensor(grid_size).cuda()
        box_wh = torch.exp(box_wh) * torch.FloatTensor(anchors).cuda()

        box_x1y1 = box_xy - box_wh / 2
        box_x2y2 = box_xy + box_wh / 2
        bbox = torch.cat([box_x1y1, box_x2y2], axis=-1)

        return bbox, objectness, class_probs, pred_box

    def _broadcast_iou(self, box_1, box_2):
        # box_1: (..., (x1, y1, x2, y2))
        # box_2: (N, (x1, y1, x2, y2))

        # broadcast boxes
        box_1 = torch.unsqueeze(box_1, -2).cuda()
        box_2 = torch.unsqueeze(box_2, 0).cuda()

        box_1, box_2 = torch.broadcast_tensors(box_1, box_2)
        box_1 = box_1.cuda()
        box_2 = box_2.cuda()

        int_w = torch.minimum(box_1[..., 2], box_2[..., 2]) - torch.maximum(box_1[..., 0], box_2[..., 0])
        int_w = torch.maximum(int_w.cuda(), torch.FloatTensor([0]).expand_as(int_w).cuda())[0]

        int_h = torch.minimum(box_1[..., 3], box_2[..., 3]) - torch.maximum(box_1[..., 1], box_2[..., 1])
        int_h = torch.maximum(int_h.cuda(), torch.FloatTensor([0]).expand_as(int_h).cuda())[0]
        int_area = int_w * int_h

        box_1_area = (box_1[..., 2] - box_1[..., 0]) * (box_1[..., 3] - box_1[..., 1])
        box_2_area = (box_2[..., 2] - box_2[..., 0]) * (box_2[..., 3] - box_2[..., 1])

        iou = int_area / (box_1_area + box_2_area - int_area)

        return iou

    def forward(self, y_pred, y_true):
        batch_size, grid_size, grid_size, n_anchors = y_pred.size()[:4]

        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, pred_xywh = self.__refine_output__(y_pred,
                                                                           self.anchor_boxes,
                                                                           self.n_classes)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, ...cls))
        true_box, true_obj, true_class_idx = torch.split(y_true, (4, 1, self.n_classes), dim=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. inverting the pred box equations
        grid = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size))
        grid = torch.unsqueeze(torch.stack(grid, axis=-1), axis=2).cuda()
        true_xy = true_xy * grid_size - grid
        true_wh = torch.log(true_wh / torch.FloatTensor(self.anchor_boxes).cuda())
        true_wh = torch.where(torch.isnan(true_wh), torch.zeros_like(true_wh), true_wh)
        true_wh = torch.where(torch.isinf(true_wh), torch.zeros_like(true_wh), true_wh)

        # 4. calculate all masks
        obj_mask = torch.squeeze(true_obj, -1)
        best_iou = self._broadcast_iou(pred_box, true_box[obj_mask.bool()])
        # ignore false positive when iou is over threshold
        if best_iou.size()[-1] == 0:
            ignore_mask = torch.ones((batch_size, grid_size, grid_size, n_anchors))
        else:
            best_iou = torch.max(best_iou, axis=-1)[0]
            ignore_mask = (best_iou < self.iou_threshold)
        ignore_mask = ignore_mask.cuda()

        # 5. calculate all losses
        xy_loss = obj_mask * box_loss_scale * torch.sum((true_xy - pred_xy) ** 2, axis=-1)
        wh_loss = obj_mask * box_loss_scale * torch.sum((true_wh - pred_wh) ** 2, axis=-1)
        obj_loss = self.bceloss(pred_obj.float(), true_obj.float()).item()
        obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss
        class_loss = obj_mask * self.bceloss(pred_class.float(), true_class_idx.float()).item()

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = torch.sum(xy_loss, axis=(1, 2, 3))
        wh_loss = torch.sum(wh_loss, axis=(1, 2, 3))
        obj_loss = torch.sum(obj_loss, axis=(1, 2, 3))
        class_loss = torch.sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + obj_loss + class_loss