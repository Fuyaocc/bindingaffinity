import torch
import torch.nn as nn
import torch.nn.functional as F

class ProDesign(nn.Module):
    def __init__(self,dim,device='cuda',layer_num=3):
        super().__init__()
        self.linear=nn.Linear(34,dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8,activation='gelu',batch_first=True,dim_feedforward=4*dim,norm_first=True)
        self.encode=nn.TransformerEncoder(encoder_layer,layer_num)
        self.fc=nn.Linear(dim,20)
        self.to(device)

    def forward(self,ret):
        nei_feature=ret['nei_feature']
        nei_mask=ret['nei_mask'].bool()
        x=self.linear(nei_feature)
        x=self.encode(x,src_key_padding_mask=~nei_mask)
        x=x.masked_fill(~nei_mask[...,None],0)
        # x=x.sum(1)/(nei_mask.sum(-1,keepdim=True)+1e-8)
        x=x.sum(1)/nei_mask.sum(-1,keepdim=True).clamp_min(1)
        return x
        # x=self.fc(x)
        # return x

    def loss(self,preds,ret):
        loss=F.cross_entropy(preds,ret['label'])
        # loss=loss*(ret['nei_mask'].sum(-1)>1)
        # return loss.mean()
        return loss

    def accuracy(self,preds,ret):
        acc = torch.eq(torch.argmax(preds, -1), ret['label']).float().mean()
        return acc
