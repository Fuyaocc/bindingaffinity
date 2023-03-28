import torch
import torch.nn as nn
import torch.nn.functional as F

class AffinityNet(nn.Module):
    def __init__(self,dim=1280,device='cuda:0'):
        super().__init__()

        self.conv1ds=nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=dim,out_channels=512,kernel_size=3,stride=1),#(32,1280,998)
                nn.ReLU(),
                nn.MaxPool1d(2)
                ),#(32,512,499)
            nn.Sequential(
                nn.Conv1d(in_channels=512,out_channels=256,kernel_size=3,stride=1),#(32,512,499)
                nn.ReLU(),
                nn.MaxPool1d(2)
                ),
            nn.Sequential(
                nn.Conv1d(in_channels=256,out_channels=128,kernel_size=3,stride=1),#(32,256,248)
                nn.ReLU(),
                nn.MaxPool1d(2)
                )
            # nn.Sequential(
            #     nn.Conv1d(in_channels=128,out_channels=64,kernel_size=3,stride=1),#(32,128,123)
            #     nn.ReLU(),
            #     nn.MaxPool1d(2)
            #     ),
        ])
        self.grus=nn.ModuleList([
            nn.GRU(512, 256, batch_first=True,bidirectional=True),
            nn.GRU(256, 128,batch_first=True,bidirectional=True),
            nn.GRU(128, 32,batch_first=True,bidirectional=True)
            # nn.GRU(64, 32,batch_first=True,bidirectional=True)
        ])
        
        self.avgpool1d= nn.AdaptiveAvgPool1d(1)
        self.fc=nn.Sequential(
            nn.Linear(1280,100,bias=True),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(100,1)
        )
        self.to(device)

    def forward(self,x1,x2,num_layers):#(32,1000,1280)
        x1=x1.to(torch.float32)
        x2=x2.to(torch.float32)
        for i in range(num_layers):
            x1=x1.permute(0, 2, 1)
            x2=x2.permute(0, 2, 1)
            x1=self.conv1ds[i](x1)
            x2=self.conv1ds[i](x2)
            x1=x1.permute(0, 2, 1)
            x2=x2.permute(0, 2, 1)
            x1,h_1=self.grus[i](x1)
            x2,h_2=self.grus[i](x2)
        #(batch_size,seq_len,embedding_len)
        x1=x1.permute(0, 2, 1)
        x2=x2.permute(0, 2, 1)
        x1=self.avgpool1d(x1)
        x2=self.avgpool1d(x2)
        # print(x1.shape)
        x1=x1.squeeze(dim=2)
        x2=x2.squeeze(dim=2)
        x3=torch.mul(x1,x2)
        # print(x3.shape)
        x3=self.fc(x3)

        return x3
