import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool,GATv2Conv,GCNConv
from torch_geometric.data import Data
import torch.nn.init as init
import logging

class Affinity_GAT(nn.Module):
    def __init__(self,num_features, hidden_channel, out_channel,device):
        super().__init__()

        self.conv1 = GATv2Conv(num_features, hidden_channel,heads=3,concat=True,edge_dim=1,dropout=0.5)
        self.drop1=nn.Dropout(0.5)
        self.conv2 = GATv2Conv(hidden_channel*3, out_channel,heads=1,concat=True,edge_dim=1,dropout=0.5)
        self.drop2=nn.Dropout(0.5)
        

    def forward(self, data, device):
        x=data.x.to(device)
        edge_index=data.edge_index.to(device)
        edge_attr=data.edge_attr.to(device)
        
        x = F.elu(self.conv1(x, edge_index,edge_attr))
        x = F.elu(self.conv2(x, edge_index,edge_attr))

        return x

class AffinityNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, mlp_in_channels,device):
        super().__init__()

        # self.gat = Affinity_GAT(in_channels, hidden_channels, out_channels,device)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.gcn_bn = nn.LayerNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

        self.bn = nn.LayerNorm(mlp_in_channels)
        self.fc1 = nn.Linear(mlp_in_channels, mlp_in_channels)
        self.bn1 = nn.LayerNorm(mlp_in_channels)
        self.fc3 = nn.Linear(mlp_in_channels, 1)

        self.noise = torch.nn.Parameter(torch.zeros(in_channels))
        self.register_parameter('noise', self.noise)
        
    def forward(self, data,mode,device):
        
        # output = self.gat(data,device)
        x1 = F.relu(self.conv1(data.x, data.edge_index))
        x1 = self.gcn_bn(F.dropout(x1, p=0.5, training=self.training))
        x1 = F.relu(self.conv2(x1, data.edge_index))
        output = x1
        x1 = F.dropout(x1, p=0.5, training=self.training)
        x1 = global_mean_pool(x1, data.batch)
        e1 = data.energy.reshape(x1.shape[0],25)
        # x1 = torch.cat((x1,e1),dim=1)
        x1 = self.bn(x1)
        x1 = F.dropout(self.bn1(F.relu(self.fc1(x1))),0.5,training=mode)
        x1 = self.fc3(x1)
        
        if mode == True:
            x2 = data.x+self.noise
            data_adv = Data(x=x2, edge_index=data.edge_index,batch=data.batch,edge_attr=data.edge_attr,y=data.y,energy=data.energy)

            x2 = F.relu(self.conv1(x2, data_adv.edge_index))
            x2 = self.gcn_bn(F.dropout(x2, p=0.5, training=self.training))
            x2 = F.relu(self.conv2(x2, data_adv.edge_index))
            output_noisy = x2
            x2 = F.dropout(x2, p=0.5, training=self.training)

            x2 = global_mean_pool(x2, data_adv.batch)
            e2 = data.energy.reshape(x2.shape[0],25)
            # x2 = torch.cat((x2,e2),dim=1)
            x2 = self.bn(x2)
            x2 = F.dropout(self.bn1(F.relu(self.fc1(x2))),0.5,training=mode)
            x2 = self.fc3(x2)
            return output, output_noisy,x1,x2
        else:
            return x1
