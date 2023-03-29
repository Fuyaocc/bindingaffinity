import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm

class AffinityNet(nn.Module):
    def __init__(self,dim=512,device='cuda:0'):
        super().__init__()

        self.encode=ProteinCNN(dim,[dim,dim,dim])
        self.encode1=ProteinCNN(dim,[dim,dim,dim])
        
        self.bcn = weight_norm(BANLayer(v_dim=dim,q_dim=dim, h_dim=dim, h_out=2) ,
                               name='h_mat', 
                               dim=None)

        self.decode = MLPDecoder(dim,dim//2)
        
        self.to(device)

    def forward(self, c1, c2, num_layers,mode="train"):
        # print(c1.shape)
        c1 = self.encode(c1)
        c2 = self.encode1(c2)
        # print('after encode:'+str(c1.shape))
        # print('after encode:'+str(c2.shape))
        f, att = self.bcn(c1, c2)
        # print(f.shape)
        if mode=="train":
            score = self.decode(f,True)
        else:
            score = self.decode(f,False)
        # print('maps:'+str(att.shape))
        return score
        # if mode == "train":
        #     return c1, c2, f, score
        # elif mode == "eval":
        #     return c1, c2, score, att
        
    
class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters):
        super(ProteinCNN, self).__init__()
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=3)
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=6)
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=9)
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = v.transpose(2, 1)
        v = self.bn1(F.leaky_relu(self.conv1(v)))
        v = self.bn2(F.leaky_relu(self.conv2(v)))
        v = self.bn3(F.leaky_relu(self.conv3(v)))
        v = v.view(v.size(0), v.size(2), -1)
        return v
    
class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=256):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, 128)
        # self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(hidden_dim, 1)


    def forward(self, x,train_flag):
        # print(x.shape)
        x = self.bn1(F.leaky_relu(self.fc1(x)))
        # x= F.dropout(x,0.5,training=train_flag)
        # print('after linear1:'+str(x.shape))
        # x = self.bn2(F.leaky_relu(self.fc2(x)))
        # x= F.dropout(x,0.1)
        x=self.fc3(x)
        return x
