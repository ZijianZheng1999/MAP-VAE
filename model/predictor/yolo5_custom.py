import torch
from torch import nn
import torchvision.ops as ops
import math


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class CBS(nn.Module):
    def __init__(self, in_channels, out_channels, k=1, s=1, p=None, g=1, d=1):
        super(CBS, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=autopad(k, p, d), groups=g, dilation=d)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=False, g=1):
        super(Bottleneck, self).__init__()
        hidden_channels = out_channels // 2
        self.cv1 = CBS(in_channels, hidden_channels, 1, 1)
        self.cv2 = CBS(hidden_channels, out_channels, 3, 1, g=g)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    def __init__(self, channel, keep_added_channel=False):
        super(C3, self).__init__()
        self.branch1 = nn.Sequential(
            CBS(channel * 2, channel, 1),
            Bottleneck(channel, channel),
            Bottleneck(channel, channel),
            Bottleneck(channel, channel),
        )
        self.branch2 = CBS(channel * 2, channel, 1)
        if keep_added_channel:
            self.output = CBS(channel * 2, channel * 2, 1)
        else:
            self.output = CBS(channel * 2, channel, 1)

    def forward(self, x):
        return self.output(torch.cat((self.branch1(x), self.branch2(x)), dim=1))


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=autopad(3), bias=False),
                nn.PixelShuffle(2),
                nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.model(x)


class DynamicModel(nn.Module):
    def __init__(self, layers_config, output_config):
        super(DynamicModel, self).__init__()
        self.layers = nn.ModuleList()
        self.connections = []

        self.output_config = output_config

        for from_idx, block_num, module, args in layers_config:
            blocks = nn.Sequential(*[module(*args) for _ in range(block_num)])
            self.layers.append(blocks)
            self.connections.append(from_idx)

    def forward(self, x):
        outputs = [x]
        for i, (layer, from_idx) in enumerate(zip(self.layers, self.connections)):
            inputs = [outputs[idx] for idx in (from_idx if isinstance(from_idx, list) else [from_idx])]
            x = torch.cat(inputs, dim=1) if len(inputs) > 1 else inputs[0]
            outputs.append(layer(x))
        return [outputs[idx] for idx in self.output_config]


class BackBoneMapping(nn.Module):
    def __init__(self, latent_dim, backbone_channel, feature_res_h, feature_res_w):
        super(BackBoneMapping, self).__init__()
        self.mapping = nn.Sequential(
            nn.Linear(latent_dim, backbone_channel * feature_res_h * feature_res_w),
            nn.ReLU(True),
            nn.Unflatten(1, (backbone_channel, feature_res_h, feature_res_w))
        )
    def forward(self, x):
        return self.mapping(x)


class YOLO5Custom(nn.Module):
    def __init__(self, latent_dim, feature_res_h, feature_res_w, num_classes, num_anchors=3, backbone_channel=1024, img_size=(375, 1224)):
        super(YOLO5Custom, self).__init__()
        self.latent_dim = latent_dim
        self.feature_res_h = feature_res_h
        self.feature_res_w = feature_res_w
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.output_dim = self.num_anchors * (5 + self.num_classes)
        self.img_size = img_size

        self.backbone_channel = backbone_channel

        # [(from [if is list, concat dim 1], block_num, module, args)]
        self.layer_config = [
            # 1: map the latent space to feature map
            (-1, 1, BackBoneMapping, (latent_dim, backbone_channel, feature_res_h, feature_res_w)),
            # 2: upsample, reduce half channel
            (-1, 1, UpSample, (backbone_channel, backbone_channel // 2)),
            # 3: upsample, reduce half channel
            (-1, 1, UpSample, (backbone_channel // 2, backbone_channel // 4)),
            # 4/5: CBS, split the backbone output from backbone_channel into 2 parts
            (1, 1, CBS, (backbone_channel, backbone_channel // 2, 1, 1)),
            (1, 1, CBS, (backbone_channel, backbone_channel // 2, 1, 1)),
            # 6: upsample
            (-1, 1, UpSample, (backbone_channel // 2, backbone_channel // 2)),
            # 7: C3 with concat input
            ([-1, 2], 1, C3, (backbone_channel // 2,)),
            # 8/9: CBS, split the channels into 2 parts
            (7, 1, CBS, (backbone_channel // 2, backbone_channel // 4, 1, 1)),
            (7, 1, CBS, (backbone_channel // 2, backbone_channel // 4, 1, 1)),
            # 10: upsample
            (-1, 1, UpSample, (backbone_channel // 4, backbone_channel // 4)),
            # 11: C3 with concat input
            ([-1, 3], 1, C3, (backbone_channel // 4,)),
            # 12/13: parallel the channels into 2 parts
            (11, 1, CBS, (backbone_channel // 4, self.output_dim, 1, 1)),
            (11, 1, CBS, (backbone_channel // 4, backbone_channel // 4, 3, 2, 1)),
            # 14: C3 with concat input
            ([-1, 8], 1, C3, (backbone_channel // 4, True)),
            # 15/16: parallel the channels into 2 parts
            (14, 1, CBS, (backbone_channel // 2, self.output_dim, 1, 1)),
            (14, 1, CBS, (backbone_channel // 2, backbone_channel // 2, 3, 2, 1)),
            # 17: C3 with concat input
            ([-1, 4], 1, C3, (backbone_channel // 2, True)),
            # 18: output layer
            (-1, 1, CBS, (backbone_channel, self.output_dim, 1, 1))
        ]
        self.output_config = [12, 15, 18]

        self.model = DynamicModel(self.layer_config, self.output_config)

    def forward(self, x):
        pred_list = self.model(x)       # [B, Channel, H, W]
        for i, pred in enumerate(pred_list):
            if i == 0:
                h = self.feature_res_h * 4
                w = self.feature_res_w * 4
            elif i == 1:
                h = self.feature_res_h * 2
                w = self.feature_res_w * 2
            else:
                h = self.feature_res_h
                w = self.feature_res_w
            pred_list[i] = pred.view(pred.shape[0], self.num_anchors, 5 + self.num_classes, h, w)
        # [B, A, 5 + C, H, W]
        return pred_list

    def decode_prediction(self, predicts, anchors, conf_threshold=0.25) -> list:
        # input shape: [B, A, 5 + C, H, W]
        batch_size, _, _, h, w = predicts.shape

        # get the prediction values
        tx, ty, tw, th = predicts[:, :, 0, :, :], predicts[:, :, 1, :, :], predicts[:, :, 2, :, :], predicts[:, :, 3, :, :]
        obj_conf = torch.sigmoid(predicts[:, :, 4, :, :])
        cls_conf = torch.softmax(predicts[:, :, 5:, :, :], dim=2)

        # get the grid
        grid_x, grid_y = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')
        grid_x, grid_y = grid_x.to(predicts.device), grid_y.to(predicts.device)
        # cx, cy: [B, A, H, W]

        # get the center coord
        stride_h = self.img_size[0] // h
        stride_w = self.img_size[1] // w
        cx = (torch.sigmoid(tx) + grid_x.unsqueeze(0).unsqueeze(0)) * stride_w
        cy = (torch.sigmoid(ty) + grid_y.unsqueeze(0).unsqueeze(0)) * stride_h
        # tw, th: [B, A, H, W]

        # get the w, h of the bbox
        anchor_w, anchor_h = anchors[:, 0], anchors[:, 1]
        w = anchor_w.view(1, self.num_anchors, 1, 1) * torch.exp(tw)
        h = anchor_h.view(1, self.num_anchors, 1, 1) * torch.exp(th)
        # w, h: [B, A, H, W]

        # get the bbox coord
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        # x1, y1, x2, y2: [B, A, H, W]

        # get the max class conf, class and overall score
        max_cls_conf, class_id = torch.max(cls_conf, dim=2)
        score = obj_conf * max_cls_conf
        # score: [B, A, H, W]

        # select the bbox with score > threshold
        mask = score > conf_threshold
        batch_result = []
        for batch in range(batch_size):
            selected_x1 = x1[batch][mask[batch]]
            selected_y1 = y1[batch][mask[batch]]
            selected_x2 = x2[batch][mask[batch]]
            selected_y2 = y2[batch][mask[batch]]
            selected_score = score[batch][mask[batch]]
            selected_class_id = class_id[batch][mask[batch]]
            result = torch.stack([selected_x1, selected_y1, selected_x2, selected_y2, selected_score, selected_class_id], dim=1)
            batch_result.append(result)
            # result: [N(selected object), 6(x1, y1, x2, y2, score, class_id)]
        return batch_result

    @staticmethod
    def apply_nms(predicts, iou_threshold=0.5):
        if predicts.numel() == 0:
            return predicts

        boxes = predicts[:, :4]
        scores = predicts[:, 4]

        keep = ops.nms(boxes, scores, iou_threshold)
        return predicts[keep]

    def post_process(self, predicts, anchors):
        batch_size = predicts[0].shape[0]
        P3, P4, P5 = predicts

        P3_results = self.decode_prediction(P3, anchors)
        P4_results = self.decode_prediction(P4, anchors)
        P5_results = self.decode_prediction(P5, anchors)

        result_list = []
        for b in range(batch_size):
            concat_results = torch.cat([P3_results[b], P4_results[b], P5_results[b]], dim=0)
            selected_results = self.apply_nms(concat_results)
            result_list.append(selected_results)

        return result_list

# https://github.com/ultralytics/yolov5/blob/master/utils/loss.py

def smooth_BCE(eps=0.1):
    """Returns label smoothing BCE targets for reducing overfitting; pos: `1.0 - 0.5*eps`, neg: `0.5*eps`. For details see https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441."""
    return 1.0 - 0.5 * eps, 0.5 * eps

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculates IoU, GIoU, DIoU, or CIoU between two boxes, supporting xywh/xyxy formats.

    Input shapes are box1(1,4) to box2(n,4).
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

class BCEBlurWithLogitsLoss(nn.Module):
    """Modified BCEWithLogitsLoss to reduce missing label effects in YOLOv5 training with optional alpha smoothing."""

    def __init__(self, alpha=0.05):
        """Initializes a modified BCEWithLogitsLoss with reduced missing label effects, taking optional alpha smoothing
        parameter.
        """
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        """Computes modified BCE loss for YOLOv5 with reduced missing label effects, taking pred and true tensors,
        returns mean loss.
        """
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    """Applies focal loss to address class imbalance by modifying BCEWithLogitsLoss with gamma and alpha parameters."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes FocalLoss with specified loss function, gamma, and alpha values; modifies loss reduction to
        'none'.
        """
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Calculates the focal loss between predicted and true labels using a modified BCEWithLogitsLoss."""
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    """Implements Quality Focal Loss to address class imbalance by modulating loss based on prediction confidence."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes Quality Focal Loss with given loss function, gamma, alpha; modifies reduction to 'none'."""
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Computes the focal loss between `pred` and `true` using BCEWithLogitsLoss, adjusting for imbalance with
        `gamma` and `alpha`.
        """
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss



class YoloLoss:
    # input shape: [B, A, H, W, 5 + C]
    """Computes the total loss for YOLOv5 model predictions, including classification, box, and objectness losses."""

    sort_obj_iou = False

    # Compute losses
    def __init__(self, device, class_num, anchors, autobalance=False):
        """Initializes ComputeLoss with model and autobalance option, autobalances losses if True."""

        # define my hyperparameters
        h = {
            "cls_pw": 1.0,  # cls BCELoss positive_weight
            "obj_pw": 1.0,  # obj BCELoss positive_weight
            "label_smoothing": 0.0,  # label smoothing epsilon
            "fl_gamma": 0.0,  # focal loss gamma
        }

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        self.balance = {3: [4.0, 1.0, 0.4]}.get(3, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list([8, 16, 32]).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = 3             # number of anchors
        self.nc = class_num     # number of classes
        self.nl = 3             # number of layers
        self.anchors = anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        # p shape: [B, A, 5 + C, H, W]
        # reshape to [B, A, H, W, 5 + C]
        p = [x.permute(0, 1, 3, 4, 2).contiguous() for x in p]
        """Performs forward pass, calculating class, box, and object loss for given predictions and targets."""
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            if n := b.shape[0]:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        """Prepares model targets from input targets (image,class,x,y,w,h) for loss computation, returning class, box,
        indices, and anchors.
        """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=self.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                # j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
                j = torch.max(r, 1 / r).max(2)[0] < 4.0
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

