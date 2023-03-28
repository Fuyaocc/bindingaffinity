import torch
import torch.nn as nn
import torch.nn.functional as F
from prodesign.model.prodesign import ProDesign

class Predictor(nn.Module):
    def __init__(self,dim,device):
        super(Predictor,self).__init__()
        self.prodesign = ProDesign(dim=dim,device=device)
        self.prodesign.fc = nn.Sequential(nn.Linear(256,20), nn.Softmax())
        self.to(device)
        self.device = device

    def forward(self,ret):
        x = self.prodesign(ret)
        return x

    def loss(self,preds,ret):
        loss=F.cross_entropy(preds,ret['label'],reduction='none')
        loss=loss*(ret['nei_mask'].sum(-1)>1)
        return loss.mean()

    def accuracy(self, preds, ret):
        acc = torch.eq(torch.argmax(preds, -1), ret['label']).float().mean()
        return acc

    def forward_onlyprodesignle(self,ret):
        x = self.prodesign(ret)
        return x