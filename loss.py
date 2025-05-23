#-*-coding:utf-8-*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cal_utils as utils
def calc_label_sim(label_1, label_2):
    Sim = label_1.float().mm(label_2.float().t())
    return Sim

def compute_od(A, B):
    A2 = (A ** 2).sum(dim=1).reshape((-1, 1))
    B2 = (B ** 2).sum(dim=1)
    D = A2 + B2 - 2 * A.mm(B.t())
    return D.sqrt()

def feature_alignment(c_labels, x, y):
        od = compute_od(x, y)
        loss1 =  ((x - y)**2).sum(1).sqrt().mean()
        k=1
        loss2 = ((- od * (1-c_labels)).topk(k=k)[0] + ( - od * (1-c_labels)).topk(k=k)[0] ).mean()
        return loss1 + loss2

def contrastive_ctriterion(fea, tau=1.):
        batch_size = fea[0].shape[0]
        all_fea = torch.cat(fea)
        sim = all_fea.mm(all_fea.t())
        n_view = 2
        sim = (sim / tau).exp()
        sim = sim - sim.diag().diag()
        sim_sum1 = sum([sim[:, v * batch_size: (v + 1) * batch_size] for v in range(n_view)])
        diag1 = torch.cat([sim_sum1[v * batch_size: (v + 1) * batch_size].diag() for v in range(n_view)])
        loss1 = -(diag1 / sim.sum(1)).log().mean()

        sim_sum2 = sum([sim[v * batch_size: (v + 1) * batch_size] for v in range(n_view)])
        diag2 = torch.cat([sim_sum2[:, v * batch_size: (v + 1) * batch_size].diag() for v in range(n_view)])
        loss2 = -(diag2 / sim.sum(1)).log().mean()
        return loss1 + loss2
def ce_loss(pred, labels):
    loss = - (labels * pred.log()).sum(1).mean()
    return loss
def discrimination_loss(x, y):
    cos = lambda x, y: x.mm(y.t()) / ((x ** 2).sum(1, keepdim=True).sqrt().mm((y ** 2).sum(1, keepdim=True).sqrt().t())).clamp(min=1e-6)
    # sim = 1 - cos(x, y)

    theta12 = cos(x, y)
    theta11 = cos(x, x)
    theta22 = cos(y, y)

    loss1 = ((theta11 - theta22)**2).mean()   # 模态内
    loss2 = ((theta12-theta12.t())**2).mean() # 模态间 
    return (loss1 + loss2)

def gce_loss(pred, labels, q):
    mae = (1. - torch.sum(labels.float() * pred, dim=1)**q).div(q)
    return mae.mean()

def category_loss(pred, labels):
    p = F.softmax(pred, dim=1)
    term = torch.tensor(0.0).cuda()
    for i in range(p.shape[0]):
        term -= torch.sum(labels[i] * (p[i].log()))
    loss = term
    return loss/p.shape[0]
def feature_augmentation(features, labels, gamma1):
    lam = gamma1
    index = np.arange(features.shape[0])
    np.random.shuffle(index)
    random_features = features[index]
    random_labels = labels[index]
    new_features = lam * features + (1 - lam) * random_features
    new_labels =  lam * labels + (1 - lam)* random_labels
    return new_features, new_labels
class LossModule(torch.nn.Module):
    def __init__(self, alpha, eta, config, **kwargs):
        torch.nn.Module.__init__(self)
        # self.L = W.T
        # tmp = torch.mm(self.proxies, W)
        self.alpha = alpha
        self.eta = eta
        self.config = config
    def forward(self, features, labels, predicts, W, epoch):
        label_emb_features = torch.mm(labels, W)
        term1 = contrastive_ctriterion([features, features])
        # term1 = discrimination_loss(label_emb_features, features)
        # q = 2
        q = min(1., self.config.q * epoch)
        term2 = gce_loss(predicts, labels, q)
        # term2 = ce_loss(predicts, labels)
        return self.alpha * term1 + self.eta * term2
    
