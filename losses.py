import torch
import torch.nn as nn
import torch.nn.functional as F

# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from torch.nn.modules.loss import CrossEntropyLoss

def Binary_dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss

def kl_loss(inputs, targets, ep=1e-8):
    kl_loss=nn.KLDivLoss(reduction='mean')
    consist_loss = kl_loss(torch.log(inputs+ep), targets)
    return consist_loss

def soft_ce_loss(inputs, target, ep=1e-8):
    logprobs = torch.log(inputs+ep)
    return  torch.mean(-(target[:,0,...]*logprobs[:,0,...]+target[:,1,...]*logprobs[:,1,...]))

def mse_loss(input1, input2):
    return torch.mean((input1 - input2)**2)

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 
    
def entropy_map(a, dim):
    em = - torch.sum(a * torch.log2(a + 1e-10), dim=dim)
    return em

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersection = torch.sum(score * target)
        union = torch.sum(score * score) + torch.sum(target * target) + smooth
        loss = 1 - intersection / union
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

import torch
from torch.nn import functional as F
import torch.nn as nn
import contextlib

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def Binary_dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss


### 

def loss_fn(x, y):
    x =  torch.nn.functional.normalize(x, dim=-1, p=2)
    y =  torch.nn.functional.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)
    
def consist_loss(inputs, targets):
        """
        Consistency regularization between two augmented views
        """
        loss = (1.0 - F.cosine_similarity(inputs, targets, dim=1)).mean()

        return loss


def discrepancy_calc(v1, v2):
    """
    dis_loss for two different classifiers
    input : v1,v2
    output : discrepancy
    """
    # print(v1.dim())
    assert v1.dim() == 4
    assert v2.dim() == 4
    n, c, h, w = v1.size()
    inner = torch.mul(v1, v2)
    v1 = v1.permute(2, 3, 1, 0)
    v2 = v2.permute(2, 3, 0, 1)
    mul = v1.matmul(v2)
    mul = mul.permute(2, 3, 0, 1)
    dis = torch.sum(mul) - torch.sum(inner)
    dis = dis / (h * w)
    return dis

def discrepancy_calc_3d(v1, v2):
    """
    dis_loss for two different classifiers
    input : v1,v2
    output : discrepancy
    """
    # print(v1.shape)  # torch.Size([4, 2, 96, 96, 96])
    assert v1.dim() == 5
    assert v2.dim() == 5
    n, c, h, w, v = v1.size()
    inner = torch.mul(v1, v2)
    v1 = v1.permute(2, 3, 4, 1, 0)
    v2 = v2.permute(2, 3, 4, 0, 1)
    mul = v1.matmul(v2)                ############### 相关性矩阵
    mul = mul.permute(2, 3, 4, 0, 1)
    dis = torch.sum(mul) - torch.sum(inner)
    dis = dis / (h * w * v)
    return dis


class softDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(softDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target):
        # print(inputs.shape, target.shape)
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice
        return loss / self.n_classes


def semi_cbc_loss(inputs, targets,
                  threshold=0.65,
                  neg_threshold=0.1,
                  conf_mask=True):
    if not conf_mask:
        raise NotImplementedError
    targets_prob = F.softmax(targets, dim=1)
    # for positive
    pos_weight = targets_prob.max(1)[0]
    pos_mask = (pos_weight >= threshold)
    # for negative
    neg_weight = targets_prob.min(1)[0]
    neg_mask = (neg_weight < neg_threshold)
    y_tilde = torch.argmax(targets, dim=1)
    if not torch.any(pos_mask):
        positive_loss_mat = torch.tensor([.0], device=targets.device)
    else:
        positive_loss_mat = F.nll_loss(torch.log_softmax(inputs, dim=1),
                                       y_tilde, reduction="none")
        positive_loss_mat = positive_loss_mat * pos_weight
        positive_loss_mat = positive_loss_mat[pos_mask]

    if not torch.any(neg_mask):
        negative_loss_mat = torch.tensor([.0], device=targets.device)
    else:
        inverse_prob = torch.clamp(1 - F.softmax(inputs, dim=1), min=1e-6, max=1.0)
        negative_loss_mat = F.nll_loss(inverse_prob.log(), (1 - y_tilde), reduction="none")
        negative_loss_mat = negative_loss_mat * neg_weight
        negative_loss_mat = negative_loss_mat[neg_mask]

    return positive_loss_mat.mean() + negative_loss_mat.mean(), None


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    # pdb.set_trace()
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8  ###2-p length of vector
    return d


def scc_loss(cos_sim,tau,lb_center_12_bg,lb_center_12_la, un_center_12_bg, un_center_12_la):
    loss_intra_bg = torch.exp((cos_sim(lb_center_12_bg, un_center_12_bg))/tau)
    loss_intra_la = torch.exp((cos_sim(lb_center_12_la, un_center_12_la))/tau)
    loss_inter_bg_la = torch.exp((cos_sim(lb_center_12_bg, un_center_12_la))/tau)
    loss_inter_la_bg = torch.exp((cos_sim(lb_center_12_la, un_center_12_bg))/tau)
    loss_contrast_bg = -torch.log(loss_intra_bg)+torch.log(loss_inter_bg_la)
    loss_contrast_la = -torch.log(loss_intra_la)+torch.log(loss_inter_la_bg)
    loss_contrast = torch.mean(loss_contrast_bg+loss_contrast_la)
    return loss_contrast


def semi_cbc_loss(inputs, targets,
                  threshold=0.6,
                  neg_threshold=0.3,
                  conf_mask=True):
    if not conf_mask:
        raise NotImplementedError
    targets_prob = F.softmax(targets, dim=1)
    # for positive
    pos_weight = targets_prob.max(1)[0]
    pos_mask = (pos_weight >= threshold)
    # for negative
    neg_weight = targets_prob.min(1)[0]
    neg_mask = (neg_weight < neg_threshold)
    y_tilde = torch.argmax(targets, dim=1)
    if not torch.any(pos_mask):
        positive_loss_mat = torch.tensor([.0], device=targets.device)
    else:
        positive_loss_mat = F.nll_loss(torch.log_softmax(inputs, dim=1),
                                       y_tilde, reduction="none")
        positive_loss_mat = positive_loss_mat * pos_weight
        positive_loss_mat = positive_loss_mat[pos_mask]

    if not torch.any(neg_mask):
        negative_loss_mat = torch.tensor([.0], device=targets.device)
    else:
        inverse_prob = torch.clamp(1 - F.softmax(inputs, dim=1), min=1e-6, max=1.0)
        negative_loss_mat = F.nll_loss(inverse_prob.log(), (1 - y_tilde), reduction="none")
        negative_loss_mat = negative_loss_mat * neg_weight
        negative_loss_mat = negative_loss_mat[neg_mask]

    return positive_loss_mat.mean() + negative_loss_mat.mean()

class VAT2d_v1(nn.Module):

    def __init__(self, xi=10.0, epi=6.0, ip=1):
        super(VAT2d_v1, self).__init__()
        self.xi = xi
        self.epi = epi
        self.ip = ip
        self.loss = softDiceLoss(4)

    def forward(self, model, x):
        with torch.no_grad():
            pred = [F.softmax(model(x)[i], dim=1) for i in range(2)]

        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)
        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_(True)
                pred_hat = model(x + self.xi * d)#[index]         ########## get by student

                logp_hat = [F.softmax(pred_hat, dim=1) for pred_hat in pred_hat]
                adv_distance = self.loss(logp_hat[0], pred[1])
                adv_distance = sum(adv_distance)
                adv_distance.backward()
                # adv_distance[0].backward()
                # adv_distance[1].backward()
                # adv_distance[2].backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            r_adv = d * self.epi
            pred_hat = model(x + r_adv)
            logp_hat = [F.softmax(pred_hat, dim=1) for pred_hat in pred_hat]
            lds = [self.loss(logp_hat, pred) for (logp_hat, pred) in zip(logp_hat, pred)]
            lds = sum(lds)
        return lds

class VAT2d_v2(nn.Module):

    def __init__(self, xi=10.0, epi=6.0, ip=1):
        super(VAT2d_v2, self).__init__()
        self.xi = xi
        self.epi = epi
        self.ip = ip
        self.loss = softDiceLoss(4)

    def forward(self, model, x):
        with torch.no_grad():
            pred = [F.softmax(model(x)[i], dim=1) for i in range(3)]

        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)
        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_(True)
                pred_hat = model(x + self.xi * d)#[index]         ########## get by student

                logp_hat = [F.softmax(pred_hat, dim=1) for pred_hat in pred_hat]
                adv_distance = [self.loss(logp_hat, pred) for (logp_hat, pred) in zip(logp_hat, pred)]
                adv_distance = sum(adv_distance)
                adv_distance.backward()
                # adv_distance[0].backward()
                # adv_distance[1].backward()
                # adv_distance[2].backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            r_adv = d * self.epi
            pred_hat = model(x + r_adv)
            logp_hat = [F.softmax(pred_hat, dim=1) for pred_hat in pred_hat]
            lds = [self.loss(logp_hat, pred) for (logp_hat, pred) in zip(logp_hat, pred) ]
            lds = sum(lds)
        return lds

##### 
class VAT2d_v2_New(nn.Module):
    def __init__(self, xi=10.0, epi=6.0, ip=1):
        super(VAT2d_v2_New, self).__init__()
        self.xi = xi
        self.epi = epi
        self.ip = ip
        self.loss = softDiceLoss(4)
    def forward(self, model, x):
        with torch.no_grad():
            # pred = [F.softmax(model(x)[i], dim=1) for i in range(2)]
            output = model(x)
            pred = [F.softmax(output[i], dim=1) for i in range(len(output))]

        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)
        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_(True)
                pred_hat = model(x + self.xi * d)#[index]         ########## get by student

                logp_hat = [F.softmax(pred_hat, dim=1) for pred_hat in pred_hat]
                adv_distance = 0
                for i in range(len(pred)):
                    for j in range(len(pred)):
                        if i != j:
                            adv_distance += self.loss(logp_hat[i], pred[j])
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            r_adv = d * self.epi
            pred_hat = model(x + r_adv)
            logp_hat = [F.softmax(pred_hat, dim=1) for pred_hat in pred_hat]
            lds = 0
            for i in range(len(pred)):
                for j in range(len(pred)):
                    if i != j:
                        lds += self.loss(logp_hat[i], pred[j])
            # lds = [self.loss(logp_hat, pred) for (logp_hat, pred) in zip(logp_hat, pred) ]
            # lds = sum(lds)
        return lds


            # for iter_num in range(args.K):
            #     x_du1 = torch.tensor(feat_u1.clone().detach().cpu().data.numpy() + 1e-3*du1.astype(np.float32), requires_grad=True)
            #     u_d_logit1 = model.module.densenet121.classifier(x_du1.cuda())
            #     u_d1_s = torch.softmax(u_d_logit1, dim=1)
            #     x_du2 = torch.tensor(feat_u2.clone().detach().cpu().data.numpy() + 1e-3*du2.astype(np.float32), requires_grad=True)
            #     u_d_logit2 = model.module.densenet121.classifier(x_du2.cuda())
            #     u_d2_s = torch.softmax(u_d_logit2, dim=1)
                
            #     cls_loss_ud = F.kl_div(u_d2_s.log(), unc_o_s1.detach(), reduction='batchmean') + F.kl_div(u_d1_s.log(), unc_o_s2.detach(), reduction='batchmean')
            #     cls_loss_ud.backward(retain_graph=True)
            #     du1 = x_du1.grad
            #     du2 = x_du2.grad
            #     du1 = du1.numpy()
            #     du2 = du2.numpy()
            #     du1 = normalize_l2(du1)
            #     du2 = normalize_l2(du2)


##### 在数据层面上实现 MVAT
class VAT2d_v2_New_Data(nn.Module):
    def __init__(self, xi=10.0, epi=6.0, ip=1):
        super(VAT2d_v2_New_Data, self).__init__()
        self.xi = xi
        self.epi = epi
        self.ip = ip
        self.loss = softDiceLoss(4)
    def forward(self, model, x):
        with torch.no_grad():
            # pred = [F.softmax(model(x)[i], dim=1) for i in range(2)]
            output, _, _ = model(x)
            pred = [F.softmax(output[i], dim=1) for i in range(len(output))]

        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)
        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_(True)
                pred_hat, _, _ = model(x + self.xi * d)#[index]         ########## get by student

                logp_hat = [F.softmax(pred_hat, dim=1) for pred_hat in pred_hat]
                adv_distance = 0
                for i in range(len(pred)):
                    for j in range(len(pred)):
                        if i != j:
                            adv_distance += self.loss(logp_hat[i], pred[j])
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            r_adv = d * self.epi
            pred_hat, _, _ = model(x + r_adv)
            logp_hat = [F.softmax(pred_hat, dim=1) for pred_hat in pred_hat]
            lds = 0
            for i in range(len(pred)):
                for j in range(len(pred)):
                    if i != j:
                        lds += self.loss(logp_hat[i], pred[j])
            # lds = [self.loss(logp_hat, pred) for (logp_hat, pred) in zip(logp_hat, pred) ]
            # lds = sum(lds)
        return lds
    
    
    
##### 在特征层面上实现 MVAT
class VAT2d_v2_Feat(nn.Module):
    def __init__(self, xi=10.0, epi=6.0, ip=1):
        super(VAT2d_v2_Feat, self).__init__()
        self.xi = xi
        self.epi = epi
        self.ip = ip
        self.loss = softDiceLoss(4)
    def forward(self, model, x):  ### x: features 
        with torch.no_grad():
            # pred = [F.softmax(model(x)[i], dim=1) for i in range(2)]
            output1 = model.branch1.out_conv(x[0].detach())
            output2 = model.branch2.out_conv(x[1].detach())
            pred1 = F.softmax(output1, dim=1)
            pred2 = F.softmax(output2, dim=1)
            # pred1 = torch.cat([y[:labeled_bs], pred1], dim=0)
            # pred2 = torch.cat([y[:labeled_bs], pred2], dim=0)


        d1 = torch.rand(x[0].shape).sub(0.5).to(x[0].device)
        d1 = _l2_normalize(d1)

        d2 = torch.rand(x[1].shape).sub(0.5).to(x[1].device)
        d2 = _l2_normalize(d2)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d1.requires_grad_(True)
                d2.requires_grad_(True)

                pred_hat1 = model.branch1.out_conv(x[0].detach()+self.xi * d1)#[index]         ########## get by student
                logp_hat1 = F.softmax(pred_hat1, dim=1)

                pred_hat2 = model.branch2.out_conv(x[1].detach()+self.xi * d2)#[index]         ########## get by student
                logp_hat2 = F.softmax(pred_hat2, dim=1)   

                adv_distance = 0
                # for i in range(len(pred)):
                #     for j in range(len(pred)):
                #         if i != j:
                adv_distance += self.loss(logp_hat2, pred1) +  self.loss(logp_hat1, pred2)
                # adv_distance.backward(retain_graph=True)
                adv_distance.backward()
                d1 = _l2_normalize(d1.grad)
                d2 = _l2_normalize(d2.grad)
                model.branch1.out_conv.zero_grad()
                model.branch2.out_conv.zero_grad()


            r_adv1 = d1 * self.epi
            pred_hat1 = model.branch1.out_conv(x[0].detach()+r_adv1)
            logp_hat1 = F.softmax(pred_hat1, dim=1)

            r_adv2 = d2 * self.epi
            pred_hat2 = model.branch2.out_conv(x[0].detach()+r_adv2)
            logp_hat2 = F.softmax(pred_hat2, dim=1)       
            lds = 0
            lds += self.loss(logp_hat1, pred2) + self.loss(logp_hat2, pred1)
        return lds
    

##### 在特征层面上实现 MVAT
class VAT2d_v2_Feat_L(nn.Module):
    def __init__(self, xi=10.0, epi=6.0, ip=1):
        super(VAT2d_v2_Feat_L, self).__init__()
        self.xi = xi
        self.epi = epi
        self.ip = ip
        # self.loss = softDiceLoss(4)
        # self.loss1 = CrossEntropyLoss()
        self.loss2 = DiceLoss(n_classes=4)
    def forward(self, model, x, y):  ### x: features 
        # with torch.no_grad():
        #     # pred = [F.softmax(model(x)[i], dim=1) for i in range(2)]
        #     output1 = model.branch1.out_conv(x[0][labeled_bs:].detach())
        #     output2 = model.branch2.out_conv(x[1][labeled_bs:].detach())
        #     pred1 = F.softmax(output1, dim=1)
        #     pred2 = F.softmax(output2, dim=1)
            # pred1 = torch.cat([y[:labeled_bs], pred1], dim=0)
            # pred2 = torch.cat([y[:labeled_bs], pred2], dim=0)


        d1 = torch.rand(x[0].shape).sub(0.5).to(x[0].device)
        d1 = _l2_normalize(d1)

        d2 = torch.rand(x[1].shape).sub(0.5).to(x[1].device)
        d2 = _l2_normalize(d2)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d1.requires_grad_(True)
                d2.requires_grad_(True)

                pred_hat1 = model.branch1.out_conv(x[0].detach()+self.xi * d1)#[index]         ########## get by student
                logp_hat1 = F.softmax(pred_hat1, dim=1)

                pred_hat2 = model.branch2.out_conv(x[1].detach()+self.xi * d2)#[index]         ########## get by student
                logp_hat2 = F.softmax(pred_hat2, dim=1)   

                adv_distance = 0
                # for i in range(len(pred)):
                #     for j in range(len(pred)):
                #         if i != j:
                # ce_loss(y, label_batch[:labeled_bs][:].long())
                # dice_loss(y_prob, label_batch[:labeled_bs].unsqueeze(1))
                # adv_distance += self.loss1(pred_hat1, y.long()) +  self.loss1(pred_hat2, y.long())
                adv_distance += self.loss2(logp_hat1, y.unsqueeze(1)) +  self.loss2(logp_hat2, y.unsqueeze(1))
                # adv_distance.backward(retain_graph=True)
                adv_distance.backward()
                d1 = _l2_normalize(d1.grad)
                d2 = _l2_normalize(d2.grad)
                model.branch1.out_conv.zero_grad()
                model.branch2.out_conv.zero_grad()

            r_adv1 = d1 * self.epi
            pred_hat1 = model.branch1.out_conv(x[0].detach()+r_adv1)
            logp_hat1 = F.softmax(pred_hat1, dim=1)

            r_adv2 = d2 * self.epi
            pred_hat2 = model.branch2.out_conv(x[0].detach()+r_adv2)
            logp_hat2 = F.softmax(pred_hat2, dim=1)       
            lds = 0
            # lds += self.loss(logp_hat1, y) + self.loss(logp_hat2, y)
            # lds += self.loss1(pred_hat1, y.long()) +  self.loss1(pred_hat2, y.long())
            lds += self.loss2(logp_hat1, y.unsqueeze(1)) +  self.loss2(logp_hat2, y.unsqueeze(1))
            
        return lds
    

class VATLoss(nn.Module):
    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip
    def forward(self, model, ema_model, x):

        with torch.no_grad():
            pred = F.softmax(ema_model(x), dim=1)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):

            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                logp_hat = F.softmax(pred_hat, dim=1)
                adv_distance = F.mse_loss(logp_hat, pred)
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(x + r_adv)
            logp_hat = F.softmax(pred_hat, dim=1)
            lds = F.mse_loss(logp_hat, pred)

        return lds

class VAT2d_v2_MT(nn.Module):

    def __init__(self, xi=10.0, epi=6.0, ip=1):
        super(VAT2d_v2_MT, self).__init__()
        self.xi = xi
        self.epi = epi
        self.ip = ip
        self.loss = softDiceLoss(4)
    def forward(self, model, ema_model, x):
        # ema_model.eval()
        with torch.no_grad():       ##### Mean Teacher
            ema_output = ema_model(x)
            ema_pred = [F.softmax(ema_output[i], dim=1) for i in range(len(ema_output))]

        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)
        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_(True)
                pred_hat = model(x + self.xi * d)#[index]         ########## get by student
                logp_hat = [F.softmax(pred_hat, dim=1) for pred_hat in pred_hat]
                adv_distance = 0
                for i in  range(len(pred_hat)):
                    for j in range(len(pred_hat)):
                        if i != j:
                            adv_distance += self.loss(logp_hat[i], ema_pred[j])
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            r_adv = d * self.epi
            x_adv = x + r_adv
            pred_hat = model(x_adv)
            logp_hat = [F.softmax(pred_hat, dim=1) for pred_hat in pred_hat]
            lds = 0
            for i in range(len(pred_hat)):
                for j in range(len(pred_hat)):
                    if i != j:
                        lds += self.loss(logp_hat[i], ema_pred[j])
            # lds = [self.loss(logp_hat, pred) for (logp_hat, pred) in zip(logp_hat, pred)]
            # lds = sum(lds)
        return lds, x_adv

class VAT2d(nn.Module):

    def __init__(self, xi=10.0, epi=6.0, ip=1):
        super(VAT2d, self).__init__()
        self.xi = xi
        self.epi = epi
        self.ip = ip
        self.loss = softDiceLoss(4)

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x)[0], dim=1)

        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)
        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_(True)
                pred_hat = model(x + self.xi * d)
                logp_hat = F.softmax(pred_hat, dim=1)
                adv_distance = self.loss(logp_hat, pred)
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            r_adv = d * self.epi
            pred_hat = model(x + r_adv)[0]
            logp_hat = F.softmax(pred_hat, dim=1)
            lds = self.loss(logp_hat, pred)
        return lds


class VAT3d(nn.Module):

    def __init__(self, xi=10.0, epi=6.0, ip=1):
        super(VAT3d, self).__init__()
        self.xi = xi
        self.epi = epi
        self.ip = ip
        self.loss = Binary_dice_loss

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x)[0], dim=1)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)  ### initialize a random tensor between [-0.5, 0.5]
        d = _l2_normalize(d)  ### an unit vector
        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_(True)
                pred_hat = model(x + self.xi * d)[0]
                p_hat = F.softmax(pred_hat, dim=1)
                adv_distance = self.loss(p_hat, pred)
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()
            pred_hat = model(x + self.epi * d)[0]
            p_hat = F.softmax(pred_hat, dim=1)
            lds = self.loss(p_hat, pred)
        return lds


class VAT3d_V1(nn.Module):

    def __init__(self, xi=10.0, epi=6.0, ip=1):
        super(VAT3d_V1, self).__init__()
        self.xi = xi
        self.epi = epi
        self.ip = ip
        self.loss = Binary_dice_loss

    def forward(self, model, x):
        with torch.no_grad():
            # pred = F.softmax(model(x)[0], dim=1)
            output = model(x)
            pred = [F.softmax(output[i], dim=1) for i in range(len(output))]

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)  ### initialize a random tensor between [-0.5, 0.5]
        d = _l2_normalize(d)  ### an unit vector
        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_(True)
                pred_hat = model(x + self.xi * d)
                p_hat = [F.softmax(pred_hat, dim=1) for pred_hat in pred_hat ]
                adv_distance = 0
                for i in range(len(pred)):
                    for j in range(len(pred)):
                        if i != j:
                            adv_distance += self.loss(p_hat[i], pred[j])
                # adv_distance = self.loss(p_hat, pred)
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()
            pred_hat = model(x + self.epi * d)
            p_hat = [F.softmax(pred_hat, dim=1) for pred_hat in pred_hat]
            lds = 0
            for i in range(len(pred)):
                for j in range(len(pred)):
                    if i != j:
                        lds += self.loss(p_hat[i], pred[j])
            # lds = self.loss(p_hat, pred)
        return lds


##### 在特征层面上实现 MVAT
class VAT3d_v2_Feat(nn.Module):
    def __init__(self, xi=10.0, epi=6.0, ip=1):
        super(VAT3d_v2_Feat, self).__init__()
        self.xi = xi
        self.epi = epi
        self.ip = ip
        self.loss = Binary_dice_loss
    def forward(self, model, x):  ### x: features 
        with torch.no_grad():
            # pred = [F.softmax(model(x)[i], dim=1) for i in range(2)]
            output1 = model.decoder1.out_conv(x[0].detach())
            output2 = model.decoder2.out_conv(x[1].detach())
            pred1 = F.softmax(output1, dim=1)
            pred2 = F.softmax(output2, dim=1)
            # pred1 = torch.cat([y[:labeled_bs], pred1], dim=0)
            # pred2 = torch.cat([y[:labeled_bs], pred2], dim=0)

        d1 = torch.rand(x[0].shape).sub(0.5).to(x[0].device)
        d1 = _l2_normalize(d1)

        d2 = torch.rand(x[1].shape).sub(0.5).to(x[1].device)
        d2 = _l2_normalize(d2)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d1.requires_grad_(True)
                d2.requires_grad_(True)

                pred_hat1 = model.decoder1.out_conv(x[0].detach()+self.xi * d1)#[index]         ########## get by student
                logp_hat1 = F.softmax(pred_hat1, dim=1)

                pred_hat2 = model.decoder2.out_conv(x[1].detach()+self.xi * d2)#[index]         ########## get by student
                logp_hat2 = F.softmax(pred_hat2, dim=1)   

                adv_distance = 0
                # for i in range(len(pred)):
                #     for j in range(len(pred)):
                #         if i != j:
                adv_distance += self.loss(logp_hat2, pred1) +  self.loss(logp_hat1, pred2)
                # adv_distance.backward(retain_graph=True)
                adv_distance.backward()
                d1 = _l2_normalize(d1.grad)
                d2 = _l2_normalize(d2.grad)
                model.decoder1.out_conv.zero_grad()
                model.decoder2.out_conv.zero_grad()

            r_adv1 = d1 * self.epi
            pred_hat1 = model.decoder1.out_conv(x[0].detach()+r_adv1)
            logp_hat1 = F.softmax(pred_hat1, dim=1)

            r_adv2 = d2 * self.epi
            pred_hat2 = model.decoder2.out_conv(x[0].detach()+r_adv2)
            logp_hat2 = F.softmax(pred_hat2, dim=1)       
            lds = 0
            lds += self.loss(logp_hat1, pred2) + self.loss(logp_hat2, pred1)
        return lds
    

class VAT3d_V1_MT(nn.Module):

    def __init__(self, xi=10.0, epi=6.0, ip=1):
        super(VAT3d_V1_MT, self).__init__()
        self.xi = xi
        self.epi = epi
        self.ip = ip
        self.loss = Binary_dice_loss

    def forward(self, model, ema_model, x):
        with torch.no_grad():  ##### Mean Teacher
            ema_output = ema_model(x)
            ema_pred = [F.softmax(ema_output[i], dim=1) for i in range(len(ema_output))]

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)  ### initialize a random tensor between [-0.5, 0.5]
        d = _l2_normalize(d)  ### an unit vector
        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_(True)
                pred_hat = model(x + self.xi * d)
                p_hat = [F.softmax(pred_hat, dim=1) for pred_hat in pred_hat ]
                adv_distance = 0
                for i in range(len(ema_pred)):
                    for j in range(len(ema_pred)):
                        if i != j:
                            adv_distance += self.loss(p_hat[i], ema_pred[j])
                # adv_distance = self.loss(p_hat, pred)
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()
            pred_hat = model(x + self.epi * d)
            p_hat = [F.softmax(pred_hat, dim=1) for pred_hat in pred_hat]
            lds = 0
            for i in range(len(ema_pred)):
                for j in range(len(ema_pred)):
                    if i != j:
                        lds += self.loss(p_hat[i], ema_pred[j])
            # lds = self.loss(p_hat, pred)
        return lds
    

class VAT3d_V1_NO_MMT(nn.Module):

    def __init__(self, xi=10.0, epi=6.0, ip=1):
        super(VAT3d_V1_NO_MMT, self).__init__()
        self.xi = xi
        self.epi = epi
        self.ip = ip
        self.loss = Binary_dice_loss

    def forward(self, model, ema_model, x):
        with torch.no_grad():  ##### Mean Teacher
            ema_output = ema_model(x)
            ema_pred = [F.softmax(ema_output[i], dim=1) for i in range(len(ema_output))]

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)  ### initialize a random tensor between [-0.5, 0.5]
        d = _l2_normalize(d)  ### an unit vector
        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_(True)
                pred_hat = model(x + self.xi * d)
                p_hat = [F.softmax(pred_hat, dim=1) for pred_hat in pred_hat ]
                adv_distance = 0
                for i in range(len(ema_pred)):
                            adv_distance += self.loss(p_hat[i], ema_pred[i])
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()
            pred_hat = model(x + self.epi * d)
            p_hat = [F.softmax(pred_hat, dim=1) for pred_hat in pred_hat]
            lds = 0
            for i in range(len(ema_pred)):
                        lds += self.loss(p_hat[i], ema_pred[i])
            # lds = self.loss(p_hat, pred)
        return lds

class EstimatorCV():
    def __init__(self, feature_num, class_num, device):
        super(EstimatorCV, self).__init__()

        self.class_num = class_num
        self.device = device
        self.CoVariance = torch.zeros(class_num, feature_num).to(device)
        self.Ave = torch.zeros(class_num, feature_num).to(device)
        self.Amount = torch.zeros(class_num).to(device)

    def update_CV(self, features, labels):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )

        label_mask = (labels == 255).long()
        labels = ((1 - label_mask).mul(labels) + label_mask * 19).long()

        onehot = torch.zeros(N, C).cuda()
        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - \
                   ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = var_temp.pow(2).sum(0).div(Amount_CxA)

        sum_weight_CV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.Amount.view(C, 1).expand(C, A)
        )

        weight_CV[weight_CV != weight_CV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul((self.Ave - ave_CxA).pow(2))

        self.CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp
                           .mul(weight_CV)).detach() + additional_CV.detach()

        self.Ave = (self.Ave.mul(1 - weight_CV) + ave_CxA.mul(weight_CV)).detach()

        self.Amount += onehot.sum(0)


class ISDALoss(nn.Module):
    def __init__(self, feature_num, class_num, device):
        super(ISDALoss, self).__init__()

        self.estimator = EstimatorCV(feature_num, class_num + 1, device)

        self.class_num = class_num

        self.cross_entropy = nn.CrossEntropyLoss()

    def isda_aug(self, fc, features, y, labels, cv_matrix, ratio):
        label_mask = (labels == 255).long()
        labels = (1 - label_mask).mul(labels).long()

        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        weight_m = list(fc.parameters())[0].squeeze()

        NxW_ij = weight_m.expand(N, C, A)

        NxW_kj = torch.gather(NxW_ij,
                              1,
                              labels.view(N, 1, 1)
                              .expand(N, C, A))

        CV_temp = cv_matrix[labels]

        sigma2 = ratio * (weight_m - NxW_kj).pow(2).mul(
            CV_temp.view(N, 1, A).expand(N, C, A)
        ).sum(2)

        aug_result = y + 0.5 * sigma2.mul((1 - label_mask).view(N, 1).expand(N, C))

        return aug_result

    def forward(self, features, final_conv, y, target_x, ratio):
        # features = model(x)

        N, A, H, W = features.size()

        target_x = target_x.view(N, 1, target_x.size(1), target_x.size(2)).float()

        target_x = F.interpolate(target_x, size=(H, W), mode='nearest', align_corners=None)

        target_x = target_x.long().squeeze()

        C = self.class_num

        features_NHWxA = features.permute(0, 2, 3, 1).contiguous().view(N * H * W, A)

        target_x_NHW = target_x.contiguous().view(N * H * W)

        y_NHWxC = y.permute(0, 2, 3, 1).contiguous().view(N * H * W, C)

        self.estimator.update_CV(features_NHWxA.detach(), target_x_NHW)

        isda_aug_y_NHWxC = self.isda_aug(final_conv, features_NHWxA, y_NHWxC, target_x_NHW,
                                         self.estimator.CoVariance.detach(), ratio)

        isda_aug_y = isda_aug_y_NHWxC.view(N, H, W, C).permute(0, 3, 1, 2)

        return isda_aug_y


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Upsample(nn.Module):

    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        if not self.size:
            size = [int(t * self.scale_factor) for t in x.shape[-2:]]
        else:
            size = self.size
        return resize(x, size, None, self.mode, self.align_corners)

def _semantic_feature(self, preds, features, conv, gt, estimator, ratio, update=True):
    N, A, H, W = features.shape
    _, C, _, _ = preds.shape
    # pdb.set_trace()
    gt_NHW = resize(gt.detach().float(), size=(H, W), mode='nearest', align_corners=None).long().reshape(N * H * W)
    features_NHWxA = features.permute(0, 2, 3, 1).reshape(N * H * W, A)
    preds_NHWxC = preds.permute(0, 2, 3, 1).reshape(N * H * W, C)
    if update:
        with torch.no_grad():
            estimator.update(features_NHWxA.detach(), gt_NHW)
    sv_NHWxC = self._semantic_vector(conv, features_NHWxA, preds_NHWxC, gt_NHW, estimator.CoVariance.detach(), ratio)
    sv = sv_NHWxC.reshape(N, H, W, C).permute(0, 3, 1, 2)
    return sv

def _semantic_vector(self, conv, features, preds, gt, CoVariance, ratio):
    gt_mask = (gt == self.ignore_index).long()
    labels = (1 - gt_mask).mul(gt).long()

    N = features.size(0)
    A = features.size(1)
    C = preds.shape[1]

    weight_m = list(conv.parameters())[0].squeeze().detach()
    NxW_ij = weight_m.expand(N, C, A)
    NxW_kj = torch.gather(NxW_ij, 1, labels.reshape(N, 1, 1).expand(N, C, A))

    CV_temp = CoVariance[labels]
    sigma2 = ratio * (weight_m - NxW_kj).pow(2).mul(CV_temp.reshape(N, 1, A).expand(N, C, A)).sum(2)

    aug_result = preds + 0.5 * sigma2.mul((1 - gt_mask).reshape(N, 1).expand(N, C).float())

    return aug_result



########## 
# peer-based binary cross-entropy
def semi_cbc_loss_acdc(inputs, targets,
                  threshold=0.6,
                  neg_threshold=0.3,
                  conf_mask=True):
    if not conf_mask:
        raise NotImplementedError
    targets_prob = F.softmax(targets, dim=1)
    # for positive
    pos_weight = targets_prob.max(1)[0]
    pos_mask = (pos_weight >= threshold)
    # for negative
    neg_weight = targets_prob.min(1)[0]
    neg_weight = neg_weight.unsqueeze(1).repeat(1, 4, 1, 1)
    neg_mask = (neg_weight < neg_threshold)

    y_tilde = torch.argmax(targets, dim=1)
    if not torch.any(pos_mask):
        positive_loss_mat = torch.tensor([.0], device=targets.device)
    else:
        positive_loss_mat = F.nll_loss(torch.log_softmax(inputs, dim=1), y_tilde, reduction="none")
        positive_loss_mat = positive_loss_mat * pos_weight
        positive_loss_mat = positive_loss_mat[pos_mask]
    
    if not torch.any(neg_mask):
        negative_loss_mat = torch.tensor([.0], device=targets.device)
    else:
        inverse_prob = torch.clamp(1-F.softmax(inputs, dim=1), min=1e-6, max=1.0)
        inverse_tilde = ~torch.nn.functional.one_hot(y_tilde, num_classes=4).permute(0, 3, 1, 2)

        negative_loss_mat = inverse_prob.log() * inverse_tilde
        negative_loss_mat = negative_loss_mat * neg_weight
        negative_loss_mat = negative_loss_mat[neg_mask]
    
    return positive_loss_mat.mean() + negative_loss_mat.mean(), None


########## 
# peer-based binary cross-entropy
def semi_crc_loss_3d(inputs, targets,
                  threshold=0.6,
                  neg_threshold=0.3,
                  conf_mask=True):
    if not conf_mask:
        raise NotImplementedError
    targets_prob = F.softmax(targets, dim=1)
    # for positive
    pos_weight = targets_prob.max(1)[0]
    pos_mask = (pos_weight >= threshold)
    # for negative
    neg_weight = targets_prob.min(1)[0]
    neg_mask = (neg_weight < neg_threshold)
    y_tilde = torch.argmax(targets, dim=1)
    if not torch.any(pos_mask):
        positive_loss_mat = torch.tensor([.0], device=targets.device)
    else:
        positive_loss_mat = F.nll_loss(torch.log_softmax(inputs, dim=1),
                                       y_tilde, reduction="none")
        positive_loss_mat = positive_loss_mat * pos_weight
        positive_loss_mat = positive_loss_mat[pos_mask]
    
    if not torch.any(neg_mask):
        negative_loss_mat = torch.tensor([.0], device=targets.device)
    else:
        inverse_prob = torch.clamp(1-F.softmax(inputs, dim=1), min=1e-6, max=1.0)
        negative_loss_mat = F.nll_loss(inverse_prob.log(), (1-y_tilde), reduction="none")
        negative_loss_mat = negative_loss_mat * neg_weight
        negative_loss_mat = negative_loss_mat[neg_mask]
    
    return positive_loss_mat.mean() + negative_loss_mat.mean(), None