import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,global_mean_pool
from torch.nn.utils.weight_norm import weight_norm

class AffinityNet(nn.Module):
    def __init__(self,num_features, hidden_channel, out_channel,device='cuda:0'):
        super().__init__()

        self.conv1 = GCNConv(num_features, hidden_channel)
        self.conv2 = GCNConv(hidden_channel, out_channel)
        
        self.fc1 = nn.Linear(out_channel, out_channel//2)
        self.bn1 = nn.BatchNorm1d(out_channel//2)
        self.fc3 = nn.Linear(out_channel//2, 1)
        
        self.to(device)

    def forward(self, x, edge_index,batch,mode=True):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=mode)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=mode)
        x = global_mean_pool(x, batch)
        x = self.bn1(F.relu(self.fc1(x)))
        x=self.fc3(x)
        return x
