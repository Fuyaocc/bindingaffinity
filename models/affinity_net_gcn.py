import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool,GATv2Conv
from torch.nn.utils.weight_norm import weight_norm

class AffinityNet(nn.Module):
    def __init__(self,num_features, hidden_channel, out_channel,device):
        super().__init__()

        # self.conv1 = GCNConv(num_features, hidden_channel)
        # self.conv2 = GCNConv(hidden_channel, out_channel)

        self.conv1 = GATv2Conv(num_features, hidden_channel,heads=1,concat=False,edge_dim=1,dropout=0.5)
        self.drop1=nn.Dropout(0.5)
        self.conv2 = GATv2Conv(hidden_channel, out_channel,heads=1,concat=False,edge_dim=1,dropout=0.5)
        self.drop2=nn.Dropout(0.5)
        # self.conv=GCNConv(num_features,out_channel)
        
        self.fc1 = nn.Linear(out_channel, out_channel)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.fc3 = nn.Linear(out_channel, 1)
        
        self.to(device)

    def forward(self, x, edge_index, batch,edge_attr, mode=True, device="cuda:0"):
        x=x.to(device)
        edge_index=edge_index.to(device)
        batch=batch.to(device)
        edge_attr=edge_attr.to(device)
        
        x = F.elu(self.conv1(x, edge_index,edge_attr))
        # if mode == True:
        #     x = self.drop1(x)
        x = F.elu(self.conv2(x, edge_index,edge_attr))
        # x=F.relu(self.conv(x,edge_index,edge_attr))
        # if mode == True:
        #     x =self.drop2(x)
        x = global_mean_pool(x, batch)
        x = F.dropout(self.bn1(F.relu(self.fc1(x))),0.5,mode)
        x=self.fc3(x)
        return x
