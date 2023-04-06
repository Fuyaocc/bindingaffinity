import torch
import torch.nn as nn
import torch.nn.functional as F

class AffinityNet(nn.Module):
    def __init__(self,dim=768,len=180,device='cuda:0'):
        super().__init__()

        self.cnn_gru=ProteinCNN(dim,len=len,device=device)
        self.fc=nn.Sequential(
            nn.Linear(128,64,bias=True),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(64,1)
        )
        self.to(device)

    def forward(self,x1,x2,num_layers,mode="train"):#(batch_size,len,dim)
        # x1=x1.to(torch.float32)
        # x2=x2.to(torch.float32)
        # for i in range(num_layers):
        #     x1=x1.permute(0, 2, 1)
        #     x2=x2.permute(0, 2, 1)
        #     x1=self.conv1ds[i](x1)
        #     x2=self.conv1ds[i](x2)
        #     x1=x1.permute(0, 2, 1)
        #     x2=x2.permute(0, 2, 1)
        #     x1,_=self.grus[i](x1)
        #     x2,_=self.grus[i](x2)
        # #(batch_size,seq_len,embedding_len)
        # x1=x1.permute(0, 2, 1)
        # x2=x2.permute(0, 2, 1)
        # x1=self.avgpool1d(x1)
        # x2=self.avgpool1d(x2)
        # # print(x1.shape)
        # x1=x1.squeeze(dim=2)
        # x2=x2.squeeze(dim=2)
        
        x1=self.cnn_gru(x1)
        x2=self.cnn_gru(x2)
        x3=torch.mul(x1,x2)
        # print(x3.shape)
        x3=self.fc(x3)

        return x3
    
class ProteinCNN(nn.Module):
    def __init__(self,dim=768,len=180,device='cuda:0'):
        super().__init__()

        self.conv1ds=nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=dim,out_channels=dim,kernel_size=3,stride=1),#(batch_size,dim,len-2)
                nn.ReLU(),
                nn.MaxPool1d(2)
                ),
            nn.Sequential(
                nn.Conv1d(in_channels=512,out_channels=512,kernel_size=3,stride=1),#(batch_size,dim,(len-2)/2-2)
                nn.ReLU(),
                nn.MaxPool1d(2)
                ),
            nn.Sequential(
                nn.Conv1d(in_channels=256,out_channels=256,kernel_size=3,stride=1),#(batch_size,dim,((len-2)/2-2)/2-2)
                nn.ReLU(),
                nn.MaxPool1d(2)
                )
        ])
        self.grus=nn.ModuleList([
            nn.GRU(dim, 256, batch_first=True,bidirectional=True),
            nn.GRU(512, 128,batch_first=True,bidirectional=True),
            nn.GRU(256, 64,batch_first=True,bidirectional=True)
        ])
        
        self.avgpool1d= nn.AdaptiveAvgPool1d(1)
        self.to(device)

    def forward(self,x):#(batch_size,len,dim)
        x=x.to(torch.float32)
        for i in range(3):
            x=x.permute(0, 2, 1)
            x=self.conv1ds[i](x)
            x=x.permute(0, 2, 1)
            x,_=self.grus[i](x)
        #(batch_size,seq_len,embedding_len)
        x=x.permute(0, 2, 1)
        x=self.avgpool1d(x)
        # print(x1.shape)
        x=x.squeeze(dim=2)

        return x
