import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing,global_mean_pool
from torch_geometric.utils import add_self_loops, degree

class MPNN(MessagePassing):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MPNN, self).__init__(aggr='add')

        self.lin = torch.nn.Sequential(
            nn.Linear(input_dim , hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        # 添加自环边
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=x.size(0))

        # 计算归一化系数
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # 节点特征传递
        x = self.lin(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * edge_attr.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        self.mpnn = MPNN(input_dim=input_dim,hidden_dim=hidden_dim, output_dim=output_dim)
        self.lin = torch.nn.Sequential(
            nn.Linear(output_dim+25 , output_dim+25),
            # nn.LayerNorm(output_dim+25),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(output_dim+25, 1)
        )
        self.noise = nn.Parameter(torch.randn(input_dim))  # 噪声参数

    def forward(self, data,mode,device):
        x = self.mpnn(data)
        output = x
        x = F.relu(x)
        x = global_mean_pool(x, data.batch)
        e = data.energy.reshape(x.shape[0],25)
        x = torch.cat([x,e],dim=1)
        x = self.lin(x)

        # if mode == True:
        #     x2 = data.x+self.noise
        #     data_adv = Data(x=x2, edge_index=data.edge_index,batch=data.batch,edge_attr=data.edge_attr,y=data.y,energy=data.energy)
        #     x2 = self.mpnn(data_adv)
        #     output2 = x2
        #     x2 = F.relu(x2)
        #     x2 = global_mean_pool(x2, data_adv.batch)
        #     x2 = self.lin(x2)
        #     return output,output2,x,x2
        # else:
        #     return x
        
        return x