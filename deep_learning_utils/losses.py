import torch
import torch.nn.functional as F
from torch import nn


class KDLoss(nn.Module):
    def __init__(self, alpha, temperature):
        super(KDLoss, self).__init__()
        self.alpha = alpha
        self.T = temperature
        self._name = "knowledge_distillation_loss"
        
    def forward(self, outputs, teacher_outputs, targets):
        student_loss = F.log_softmax(outputs / self.T, dim=1)
        teacher_loss = F.softmax(teacher_outputs / self.T, dim=1)
        kd_loss = F.kl_div(student_loss, teacher_loss)
        kd_loss = kd_loss * self.alpha * self.T * self.T + F.cross_entropy(outputs, targets) * (1 - self.alpha)
        return kd_loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self._name = "contrastive_loss"
        
    def forward(self, output1, output2, label):
        # Compute euclidean distance
        B = label.size(0)

        output1 = output1.view(B, -1)
        output2 = output2.view(B, -1)

        label   = label.view(B)
        
        output1 = F.normalize(output1, dim=1)
        output2 = F.normalize(output2, dim=1)

        euclidean_distance = F.pairwise_distance(output1, output2)

        # Label: 1 - output of same type | 0 - output of different type
        similar = label * torch.pow(euclidean_distance, 2)
        diff = (1. - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        loss_contrastive = torch.mean(similar + diff)
        return loss_contrastive
