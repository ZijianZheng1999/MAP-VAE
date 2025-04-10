import os
import numpy as np
import math
import torch
import torchvision
from torch.utils.data import Dataset

from dataset.kitti_point_cloud import KittiPointCloud


class KittiMultiModal(Dataset):
    def __init__(self, root, transform, num_samples, G, B, C, height=375.0, width=1224.0):
        super(KittiMultiModal, self).__init__()
        self.root = root
        self.transform = transform
        self.height = height
        self.width = width
        self.G = G
        self.B = B
        self.C = C
        self.img_dataset = torchvision.datasets.Kitti(self.root, download=False, transform=self.transform)
        self.cloud_dataset = KittiPointCloud(self.root)
        self.category_mapping = {
            'Car': 0,
            'Van': 1,
            'Truck': 2,
            'Pedestrian': 3,
            'Person_sitting': 4,
            'Cyclist': 5,
            'Tram': 6,
            'Misc': 7,
            'DontCare': 8
        }
        self.category_inv_mapping = [
            'Car',
            'Van',
            'Truck',
            'Pedestrian',
            'Person_sitting',
            'Cyclist',
            'Tram',
            'Misc',
            'DontCare'
        ]

    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, idx):
        cloud = self.cloud_dataset[idx]
        img, targets = self.img_dataset[idx]
        # parse target
        # shape: [objects*{type(str), _(1), _(1), alpha, bbox(left, up, right, down), _, _, _}]
        bbox = []
        labels = []
        for target in targets:
            # target: {type(str), _(1), _(1), alpha, bbox(left, up, right, down), _, _, _}
            bbox.append(target['bbox'])
            labels.append(self.category_mapping[target['type']])
        bbox = torch.tensor(bbox)
        bbox[:, 0] /= self.width
        bbox[:, 1] /= self.height
        bbox[:, 2] /= self.width
        bbox[:, 3] /= self.height
        bbox[bbox > 1.0] = 1.0
        bbox[bbox < 0.0] = 0.0
        labels = torch.tensor(labels, dtype=torch.long)
        idx = torch.tensor([idx] * len(labels), dtype=torch.long)

        # convert the target into [idx, class, x_center, y_center, width, height]
        x_center = (bbox[:, 0] + bbox[:, 2]) / 2.0
        y_center = (bbox[:, 1] + bbox[:, 3]) / 2.0
        width = bbox[:, 2] - bbox[:, 0]
        height = bbox[:, 3] - bbox[:, 1]

        target = torch.stack([idx, labels, x_center, y_center, width, height], dim=1)

        return img, cloud, target

    def target_encoder(self, boxes, labels):

        def bbox_to_yolo(bbox, grid_size):
            xmin, ymin, xmax, ymax = bbox
            x_center = (xmin + xmax) / 2.0
            y_center = (ymin + ymax) / 2.0
            width = xmax - xmin
            height = ymax - ymin

            cell_size = 1.0 / grid_size
            col = min(grid_size - 1, max(0, int(x_center / cell_size)))
            row = min(grid_size - 1, max(0, int(y_center / cell_size)))

            x_center_cell = (x_center / cell_size) - col
            y_center_cell = (y_center / cell_size) - row

            return row, col, [x_center_cell, y_center_cell, width, height]

        grid_labels = torch.zeros((self.G, self.G, 5 * self.B + self.C), dtype=torch.float32)

        for i in range(labels.size(0)):
            bbox = boxes[i, :]
            class_idx = labels[i]
            row, col, bbox_yolo = bbox_to_yolo(bbox, self.G)
            for b in range(self.B):
                if grid_labels[row, col, b * 5] == 0:
                    grid_labels[row, col, b * 5:b * 5 + 5] = torch.tensor(bbox_yolo + [1])
                    grid_labels[row, col, self.B * 5 + class_idx] = 1
                    break

        return grid_labels

    def target_decoder(self, pred):
        '''
        pred (tensor) 1x7x7x30
        return (tensor) box[[x1,y1,x2,y2]] label[...]
        '''
        grid_num = self.G
        boxes = []
        cls_indexs = []
        probs = []
        cell_size = 1. / grid_num
        pred = pred.data
        pred = pred.squeeze(0)  # 7x7x30
        contain1 = pred[:, :, 4].unsqueeze(2)
        contain2 = pred[:, :, 9].unsqueeze(2)
        contain = torch.cat((contain1, contain2), 2)
        mask1 = contain > 0.1
        mask2 = (contain == contain.max())  # select the best contain_prob what ever it>0.9
        mask = (mask1 + mask2).gt(0)
        # min_score,min_index = torch.min(contain,2)
        for i in range(grid_num):
            for j in range(grid_num):
                for b in range(self.B):
                    # index = min_index[i,j]
                    # mask[i,j,index] = 0
                    if mask[i, j, b] == 1:
                        box = pred[i, j, b * 5:b * 5 + 4]
                        contain_prob = torch.FloatTensor([pred[i, j, b * 5 + 4]])
                        xy = torch.FloatTensor([j, i]) * cell_size  # up left of cell
                        box[:2] = box[:2] * cell_size + xy  # return cxcy relative to image
                        box_xy = torch.FloatTensor(box.size())  # convert[cx,cy,w,h] to [x1,xy1,x2,y2]
                        box_xy[:2] = box[:2] - 0.5 * box[2:]
                        box_xy[2:] = box[:2] + 0.5 * box[2:]
                        max_prob, cls_index = torch.max(pred[i, j, 10:], 0)
                        if float((contain_prob * max_prob)[0]) > 0.1:
                            boxes.append(box_xy.view(1, 4))
                            cls_indexs.append(cls_index.unsqueeze(0))
                            probs.append(contain_prob * max_prob)
        if len(boxes) == 0:
            boxes = torch.zeros((1, 4))
            probs = torch.zeros(1)
            cls_indexs = torch.zeros(1)
        else:
            boxes = torch.cat(boxes, 0)  # (n,4)
            probs = torch.cat(probs, 0)  # (n,)
            cls_indexs = torch.cat(cls_indexs, 0)  # (n,)
        keep = KittiMultiModal.nms(boxes, probs)
        return boxes[keep], cls_indexs[keep], probs[keep]

    @staticmethod
    def nms(bboxes, scores, threshold=0.5):
        '''
        bboxes(tensor) [N,4]
        scores(tensor) [N,]
        '''
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        _, order = scores.sort(0, descending=True)
        keep = []
        while order.numel() > 0:
            if order.dim() == 0:
                i = order.item()
            else:
                i = order[0]
            keep.append(i)

            if order.numel() == 1:
                break

            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])

            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            ids = (ovr <= threshold).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids + 1]
        return torch.LongTensor(keep)
