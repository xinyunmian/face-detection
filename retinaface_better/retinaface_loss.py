import torch
import torch.nn as nn
import torch.nn.functional as F
from retinaface_utils import match, log_sum_exp
from config import cfg_slimNet

################criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)###############################
class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes, overlap_thresh, neg_pos):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.negpos_ratio = neg_pos
        self.variance = [0.1, 0.2]
        self.log_vars = nn.Parameter(torch.zeros(3), requires_grad=True).cuda()

    def forward(self, predictions, priors, targets):
        loc_data, conf_data, landm_data = predictions
        priors = priors
        batch_num = loc_data.size(0)
        num_priors = (priors.size(0))

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(batch_num, num_priors, 4)
        landm_t = torch.Tensor(batch_num, num_priors, 10)
        conf_t = torch.LongTensor(batch_num, num_priors)
        for idx in range(batch_num):
            truths = targets[idx][:, :4].data
            labels = targets[idx][:, -1].data
            landms = targets[idx][:, 4:14].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)

        loc_t = loc_t.cuda()
        conf_t = conf_t.cuda()
        landm_t = landm_t.cuda()

        zeros = torch.tensor(0).cuda()
        # landm Loss (Smooth L1)
        # Shape: [batch,num_priors,10]
        pos_landm = conf_t > zeros
        num_pos_landm = pos_landm.long().sum(1, keepdim=True)
        N1 = max(num_pos_landm.data.sum().float(), 1)
        pos_landm_expand = pos_landm.unsqueeze(pos_landm.dim()).expand_as(landm_data)
        landm_p = landm_data[pos_landm_expand].view(-1, 10)
        landm_t = landm_t[pos_landm_expand].view(-1, 10)
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')

        ones = torch.tensor(1).cuda() #face with landmarks
        pos_box1 = conf_t == ones
        num_pos1 = pos_box1.long().sum(1, keepdim=True)
        neg_ones = torch.tensor(-1).cuda() #face box only
        pos_box2 = conf_t == neg_ones
        num_pos2 = pos_box2.long().sum(1, keepdim=True)
        pos_box = pos_box1 + pos_box2
        num_poss = pos_box.long().sum(1, keepdim=True)
        conf_t[pos_box] = 1

        neg_tens = torch.tensor(-10).cuda()
        pos_bg = conf_t == neg_tens
        num_pobg = pos_bg.long().sum(1, keepdim=True)
        conf_t[pos_bg] = 0

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_box_expand = pos_box.unsqueeze(pos_box.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_box_expand].view(-1, 4)
        loc_t = loc_t[pos_box_expand].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos_box.view(-1, 1)] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(batch_num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos_box.long().sum(1, keepdim=True)
        num_neg = torch.Tensor(num_pos.shape).long()
        for id in range(batch_num):
            if num_pos[id] < 1:
                num_neg[id] = 10
            else:
                num_neg[id] = self.negpos_ratio * num_pos[id]
        num_neg = num_neg.cuda()
        num_neg = torch.clamp(num_neg, max=pos_box.size(1) - 1)

        neg_pos = idx_rank < num_neg.expand_as(idx_rank)
        numneg = neg_pos.long().sum(1, keepdim=True)
        neg_pos += pos_bg
        num_negsum = neg_pos.long().sum(1, keepdim=True)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos_box.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg_pos.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos_box + neg_pos).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        if N == 1:
            N = 10
        loss_l /= N
        loss_c /= N
        loss_landm /= N1
        var = torch.exp(-self.log_vars)
        loss_sum = (loss_l * var[0] + self.log_vars[0]) + (loss_c * var[1] + self.log_vars[1]) + (loss_landm * var[2] + self.log_vars[2])
        # loss_sum = loss_l + loss_c + loss_landm

        return loss_l, loss_c, loss_landm, loss_sum


################criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)###############################
class loss_origin(nn.Module):
    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        super(loss_origin, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]
        self.log_vars = nn.Parameter(torch.zeros(3), requires_grad=True).cuda()

    def forward(self, predictions, priors, targets):
        loc_data, conf_data, landm_data = predictions
        priors = priors
        num = loc_data.size(0)
        num_priors = (priors.size(0))

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        landm_t = torch.Tensor(num, num_priors, 10)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :4].data
            labels = targets[idx][:, -1].data
            landms = targets[idx][:, 4:14].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)

        loc_t = loc_t.cuda()
        conf_t = conf_t.cuda()
        landm_t = landm_t.cuda()

        zeros = torch.tensor(0).cuda()
        # landm Loss (Smooth L1)
        # Shape: [batch,num_priors,10]
        pos1 = conf_t > zeros
        num_pos_landm = pos1.long().sum(1, keepdim=True)
        N1 = max(num_pos_landm.data.sum().float(), 1)
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
        landm_p = landm_data[pos_idx1].view(-1, 10)
        landm_t = landm_t[pos_idx1].view(-1, 10)
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')

        pos = conf_t != zeros
        conf_t[pos] = 1

        tens = torch.tensor(-10).cuda()
        pos_bg = conf_t == tens
        conf_t[pos_bg] = 0

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        lsconf = log_sum_exp(batch_conf)
        bcgat = batch_conf.gather(1, conf_t.view(-1, 1))
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        pos_bg_idx = pos_bg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx + pos_bg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg + pos_bg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        loss_landm /= N1
        var = torch.exp(-self.log_vars)
        loss_sum = (loss_l * var[0] + self.log_vars[0]) + (loss_c * var[1] + self.log_vars[1]) + (loss_landm * var[2] + self.log_vars[2])
        # loss_sum = loss_l + loss_c + loss_landm

        return loss_l, loss_c, loss_landm, loss_sum































