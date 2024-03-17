import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from torchmetrics.classification import BinaryRecall

def accuracy(preds, target, device, threshold=0.5):
    preds = (preds > threshold).float()
    accuracy = Accuracy(task="binary").to(device)
    return accuracy(preds, target)

def recall(preds, target, device, threshold=0.5):
    recall = BinaryRecall(threshold=threshold).to(device)
    return recall(preds, target.int())

def dice(target, preds, threshold=0.5, smooth=1):
    batch_size = preds.size(0)
    preds = (preds > threshold).float()
    
    preds_flat = preds.view(batch_size, -1)
    target_flat = target.view(batch_size, -1)
    
    intersection = torch.sum(torch.abs(preds_flat * target_flat), dim=1)
    sum_ = torch.sum(torch.abs(preds_flat) + torch.abs(target_flat), dim=1)
    
    metric = 2 * intersection / sum_
    return torch.mean(metric)

def jaccard_index(target, preds, threshold=0.5, smooth=1):
    batch_size = preds.size(0)
    preds = (preds > threshold).float()
    
    preds_flat = preds.view(batch_size, -1)
    target_flat = target.view(batch_size, -1)

    intersection = torch.sum(torch.abs(preds_flat * target_flat), dim=1)
    sum_ = torch.sum(torch.abs(preds_flat) + torch.abs(target_flat), dim=1)

    metric = (intersection + smooth) / (sum_ - intersection + smooth)
    return torch.mean(metric)

def dice_loss(preds, target, threshold=0.5, smooth=1):
    metric = dice(preds, target, threshold, smooth)
    return (1 - metric) * smooth
    
def jaccard_index_loss(preds, target, threshold=0.5, smooth=1):
    metric = jaccard_index(preds, target, threshold, smooth)
    return (1 - metric) * smooth

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None):
        super(DiceBCELoss, self).__init__()

    def forward(self, preds, target):
        dice_coef_loss = dice_loss(preds, target)
        bce_loss = F.binary_cross_entropy(preds, target).item()
        return dice_coef_loss + bce_loss

class DiceJacBCELoss(nn.Module):
    def __init__(self, weight=None):
        super(DiceJacBCELoss, self).__init__()

    def forward(self, preds, target, threshold=0.5, smooth=1):
        # preds = preds[:,1].float()
        preds = preds.float()
        target = target.float()
        dice_coef_loss = dice_loss(preds, target, threshold, smooth)
        jac_loss = jaccard_index_loss(preds, target, threshold, smooth)
        bce_loss = F.binary_cross_entropy(preds, target)
        return dice_coef_loss + jac_loss + bce_loss

class DiceJacNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(DiceJacNLLLoss, self).__init__()

    def forward(self, preds, target, threshold=0.5, smooth=1):
        nll_loss = F.nll_loss(preds, target)
        
        preds = F.softmax(preds, dim=1)
        preds = preds[:,1]
        
        dice_coef_loss = dice_loss(preds, target, threshold, smooth)
        jac_loss = jaccard_index_loss(preds, target, threshold, smooth)

        loss = nll_loss + dice_coef_loss + jac_loss
        return loss

# A variant on the Tversky loss that also includes the gamma modifier from Focal Loss.
ALPHA = 0.2
BETA = 0.8
GAMMA = 2

class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, preds, targets, smooth=1, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #preds = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = preds.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth) 
        #TverskyLoss = 1 - Tversky
        #only TverskyLoss can be used as well (gamma=1)
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky

BCE_RATIO = 0.2

class BFTLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, preds, targets, smooth=1, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #preds = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = preds.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth) 
        #TverskyLoss = 1 - Tversky
        #only TverskyLoss can be used as well (gamma=1)
        FocalTversky = (1 - Tversky)**gamma

        BCE = F.binary_cross_entropy(preds, target)
        
        return BCE_RATIO * BCE + (1 - BCE_RATIO) * FocalTversky
        
# Combo loss is a combination of Dice Loss and a modified Cross-Entropy function that, like Tversky loss, 
# has additional constants which penalise either false positives or false negatives more respectively.

ALPHA = 0.8 # < 0.5 penalises FP more, > 0.5 penalises FN more
CE_RATIO = 0.5 #weighted contribution of modified CE loss compared to Dice loss

class ComboLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()

    def forward(self, preds, targets, smooth=1, alpha=ALPHA, beta=BETA, eps=1e-9):
        
        #flatten label and prediction tensors
        inputs = preds.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()    
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        inputs = torch.clamp(inputs, eps, 1.0 - eps)       
        out = - (ALPHA * ((targets * torch.log(inputs)) + ((1 - ALPHA) * (1.0 - targets) * torch.log(1.0 - inputs))))
        weighted_ce = out.mean(-1)
        combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)
        
        return combo

#from torchmetrics.classification import Dice, BinaryJaccardIndex

# def dice(preds, target, threshold=0.5):
#     metric = Dice(average='micro', ignore_index=0, threshold=threshold)
#     return metric(preds, target.int())

# def jaccard_index(preds, target, threshold=0.5):
#     metric = BinaryJaccardIndex(ignore_index=0, threshold=threshold)
#     return metric(preds, target.int())
