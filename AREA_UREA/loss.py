import torch
import torch.nn as nn


def ramp_loss(pred):
    return torch.clamp(1-pred,0,2)/2

def hinge_loss(pred):
    return torch.clamp(1-pred,min=0)

def sigmoid_loss(pred):
    loss = nn.Sigmoid(-pred)
    return loss

def zero_loss(pred):
    loss = torch.ones_like(pred)
    loss[pred>0] = 0
    return loss

def multi_class_loss(pred, Y_test): # pred is n by k 
    k = pred.shape[1]
    positive_loss_matrix = hinge_loss(pred)
    negative_loss_matrix = hinge_loss(-pred)
    labeled_loss = (positive_loss_matrix*Y_test).sum(dim=-1)
    unlabeled_loss = (negative_loss_matrix*(1-Y_test)).sum(dim=-1)
    loss = labeled_loss + 1.0/(k-1)*unlabeled_loss
    return loss.mean()

