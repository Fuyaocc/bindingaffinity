import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool,GATv2Conv
from torch_geometric.data import Data


# class MyData(Data):
#     def __init__(self, x, edge_index,batch,edge_attr, y):
#         super().__init__(x=x, edge_index=edge_index, batch=batch,edge_attr=edge_attr,y=y)
#         self.x_noisy = torch.nn.Parameter(torch.zeros_like(x))

class Affinity_GAT(nn.Module):
    def __init__(self,num_features, hidden_channel, out_channel,device):
        super().__init__()

        self.conv1 = GATv2Conv(num_features, hidden_channel,heads=1,concat=False,edge_dim=1,dropout=0.5)
        self.drop1=nn.Dropout(0.5)
        self.conv2 = GATv2Conv(hidden_channel, out_channel,heads=1,concat=False,edge_dim=1,dropout=0.5)
        self.drop2=nn.Dropout(0.5)
        
        self.to(device)

    def forward(self, data, device):
        x=data.x.to(device)
        edge_index=data.edge_index.to(device)
        edge_attr=data.edge_attr.to(device)
        
        x = F.elu(self.drop1(self.conv1(x, edge_index,edge_attr)))
        x = F.elu(self.drop2(self.conv2(x, edge_index,edge_attr)))
        return x

class AffinityNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, mlp_in_channels,device):
        super().__init__()
        self.gat = Affinity_GAT(in_channels, hidden_channels, out_channels,device)
        self.noise = torch.nn.Parameter(torch.zeros(in_channels))
        self.fc1 = nn.Linear(mlp_in_channels, mlp_in_channels//2)
        self.bn1 = nn.BatchNorm1d(mlp_in_channels//2)
        self.fc3 = nn.Linear(mlp_in_channels//2, 1)
        
    def forward(self, data,mode,device):
        
        output = self.gat(data,device)

        x1 = global_mean_pool(output, data.batch)
        e1 = data.energy.reshape(x1.shape[0],25)
        # x1 = torch.cat((x1,e1),dim=1)
        x1 = F.dropout(self.bn1(F.relu(self.fc1(x1))),0.5,training=mode)
        x1 = self.fc3(x1)
        
        if mode == True:
            x_noisy = torch.randn_like(data.x)+self.noise
            data_adv = Data(x=x_noisy, edge_index=data.edge_index,batch=data.batch,edge_attr=data.edge_attr,y=data.y,energy=data.energy)
            output_noisy = self.gat(data_adv,device)
            x2 = global_mean_pool(output_noisy, data_adv.batch)
            e2 = data.energy.reshape(x2.shape[0],25)
            # x2 = torch.cat((x2,e2),dim=1)
            x2 = F.dropout(self.bn1(F.relu(self.fc1(x2))),0.5,training=mode)
            x2 = self.fc3(x2)
            return output, output_noisy,x1,x2
        else:
            return x1
