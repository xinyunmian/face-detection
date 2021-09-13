import torch
import torch.nn as nn
import numpy as np
from FaceUtils import bbox_iou

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class WingLoss(nn.Module):
    def __init__(self, w=10, e=2):
        super(WingLoss, self).__init__()
        # https://arxiv.org/pdf/1711.06753v4.pdf   Figure 5
        self.w = w
        self.e = e
        self.C = self.w - self.w * np.log(1 + self.w / self.e)

    def forward(self, x, t, sigma=1):
        weight = torch.ones_like(t)
        weight[torch.where(t==-1)] = 0
        diff = weight * (x - t)
        abs_diff = diff.abs()
        flag = (abs_diff.data < self.w).float()
        y = flag * self.w * torch.log(1 + abs_diff / self.e) + (1 - flag) * (abs_diff - self.C)
        return y.sum()

class LandmarksLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=1.0):
        super(LandmarksLoss, self).__init__()
        self.loss_fcn = WingLoss()#nn.SmoothL1Loss(reduction='sum')
        self.alpha = alpha

    def forward(self, pred, truel, mask):
        loss = self.loss_fcn(pred*mask, truel*mask)
        return loss / (torch.sum(mask) + 10e-14)


def build_FaceTargets(pre_boxes, target, anchors, wh_ratio=4, strides=[8, 16, 32, 64], not_yaml=True):
    tcls, tbox, indices, anch, landmarks, lmks_mask = [], [], [], [], [], []
    nstrides = len(pre_boxes)
    na = anchors.shape[1] #shape = 4,3,2
    nt = target.shape[0]
    gain = torch.ones(17, device=target.device)
    ai = torch.arange(na, device=target.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    target = torch.cat((target.repeat(na, 1, 1), ai[:, :, None]), 2)

    g = 0.5  # bias
    off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=target.device).float() * g

    for i in range(nstrides):
        pre_out = pre_boxes[i]
        down_stride = strides[i]

        if not_yaml:
            nbatch, nchannel, nh, nw = pre_out.shape
            pre_out = pre_out.view(nbatch, na, 16, nh, nw)
            pre_out = pre_out.permute(0, 1, 3, 4, 2)  # b 3 40 40 16

        gain[2:6] = torch.tensor(pre_out.shape)[[3, 2, 3, 2]]  # xyxy gain
        # landmarks 10
        gain[6:16] = torch.tensor(pre_out.shape)[[3, 2, 3, 2, 3, 2, 3, 2, 3, 2]]
        anchor = anchors[i, :, :]
        # anchor = torch.tensor(np.array(anchor, dtype=np.float32))
        # down_strides = down_stride * torch.ones(anchor.shape, device=target.device)
        anchor = anchor / down_stride

        tb = target * gain
        if nt:
            # Matches
            # aaa = anchor[:, None]
            r = tb[:, :, 4:6] / anchor[:, None]  # wh ratio

            # mr = torch.max(r, 1. / r)
            # mmr = mr.max(2)
            # mmr0 = mmr[0]
            # j = mmr0 < wh_ratio

            j = torch.max(r, 1. / r).max(2)[0] < wh_ratio  # compare
            tb = tb[j]  # filter

            # Offsets
            gxy = tb[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse

            # gxyt = gxy % 1.
            # gxyt = gxyt < g
            # gxyr = (gxy > 1.).T

            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            tb = tb.repeat((5, 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
        else:
            tb = target[0]
            offsets = 0

        # Define
        bimg, bclass = tb[:, :2].long().T  # image, class
        gxy = tb[:, 2:4]  # grid xy
        gwh = tb[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        # Append
        a = tb[:, 16].long()  # anchor indices
        indices.append((bimg, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
        gij = gij.float()
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchor[a])  # anchor
        tcls.append(bclass)  # class

        # landmarks
        lks = tb[:, 6:16]
        lks_mask = torch.where(lks < 0, torch.full_like(lks, 0.), torch.full_like(lks, 1.0))
        lks[:, [0, 1]] = (lks[:, [0, 1]] - gij)
        lks[:, [2, 3]] = (lks[:, [2, 3]] - gij)
        lks[:, [4, 5]] = (lks[:, [4, 5]] - gij)
        lks[:, [6, 7]] = (lks[:, [6, 7]] - gij)
        lks[:, [8, 9]] = (lks[:, [8, 9]] - gij)

        lmks_mask.append(lks_mask)
        landmarks.append(lks)
    return tcls, tbox, indices, anch, landmarks, lmks_mask

def compute_FaceLoss(outputs, label, all_anchors, wh_r=4, downs=[8, 16, 32, 64], not_yaml=True):  # predictions, targets
    # label: batchid, classid, xywh, landms, anchorid 17
    # outputs[i]: xywh, conf(iou), landms, classid 16
    device = label.device
    all_anchors = torch.tensor(np.array(all_anchors, dtype=np.float32)).to(device)
    na = all_anchors.shape[1]
    nc = 1
    lcls = torch.zeros(1, device=device)
    lbox = torch.zeros(1, device=device)
    lobj = torch.zeros(1, device=device)
    lmark = torch.zeros(1, device=device)
    tcls, tbox, indices, anchors, tlandmarks, lmks_mask = build_FaceTargets(outputs, label, all_anchors, wh_r, downs, not_yaml)

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))  # weight=model.class_weights)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
    landmarks_loss = LandmarksLoss(1.0)
    cp = 1.0
    cn = 0.0
    # Focal loss
    g = 0.0  # focal loss gamma: 1.5
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    nt = 0  # number of targets
    no = len(outputs)  # number of outputs
    balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
    for i, pi in enumerate(outputs):
        if not_yaml:
            nbatch, nchannel, nh, nw = pi.shape
            pi = pi.view(nbatch, na, 16, nh, nw)
            pi = pi.permute(0, 1, 3, 4, 2)

        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

        n = b.shape[0]  # number of targets
        if n:
            nt += n  # cumulative targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # Regression
            pxy = ps[:, :2].sigmoid() * 2. - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
            pbox = torch.cat((pxy, pwh), 1)  # predicted box
            iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
            lbox += (1.0 - iou).mean()  # iou loss

            # Objectness
            tobj[b, a, gj, gi] = iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

            # Classification
            if nc > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 15:], cn, device=device)  # targets
                t[range(n), tcls[i]] = cp
                lcls += BCEcls(ps[:, 15:], t)  # BCE

            # landmarks loss
            plandmarks = ps[:, 5:15]
            plandmarks[:, 0:2] = plandmarks[:, 0:2] * anchors[i]
            plandmarks[:, 2:4] = plandmarks[:, 2:4] * anchors[i]
            plandmarks[:, 4:6] = plandmarks[:, 4:6] * anchors[i]
            plandmarks[:, 6:8] = plandmarks[:, 6:8] * anchors[i]
            plandmarks[:, 8:10] = plandmarks[:, 8:10] * anchors[i]
            lmark += landmarks_loss(plandmarks, tlandmarks[i], lmks_mask[i])

        lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

    s = 3 / no  # output count scaling
    lbox *= 0.05 * s
    lobj *= 1.0 * s * (1.4 if no == 4 else 1.)
    lcls *= 0.5 * s
    lmark *= 0.005 * s

    bs = tobj.shape[0]  # batch size
    loss = lbox + lobj + lcls + lmark
    return loss * bs, torch.cat((lbox, lobj, lcls, lmark, loss)).detach()

























