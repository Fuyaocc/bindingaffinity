import torch
import logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
from torch_geometric.data import Data,DataLoader,Batch
from   torch.nn  import KLDivLoss
import torch.nn.functional as F
import torch_geometric as pyg

# from  torchsummary import summary

def js_div(p_output, q_output, get_softmax=True):
    kl_div = KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_output,dim=-1)
        q_output = F.softmax(q_output,dim=-1)
    log_mean_output = F.log_softmax((p_output + q_output)/2, dim=-1)
    return (kl_div(log_mean_output, p_output) + kl_div(log_mean_output, q_output))/2

def generate_adv_sample(batch,epsilon):
    graph_list=[]
    for i in range(batch.num_graphs):
        sub_data = batch[i]
        delta = torch.zeros_like(sub_data.x).uniform_(-epsilon, epsilon)
        delta.requires_grad = True
        x_adv = sub_data.x + delta        
        graph_list.append(Data(x=x_adv,edge_index=sub_data.edge_index,edge_attr=sub_data.edge_attr,y=sub_data.y))
    grapth_adv=Batch.from_data_list(graph_list)
    return grapth_adv
    

def run_train(model,dataloader,optimizer,criterion,device,i,epoch,outDir,num_layers):
    model.train()
    model.to(device)
    dataitr=iter(dataloader)
    f=open(f'./tmp/train/val{i}/train_'+str(epoch)+'.txt','w')
    prelist = []
    truelist = []
    epoch_loss=0.0
    for batch_id,data in enumerate(dataitr):
        optimizer.zero_grad()
        features=data[:-1]
        for feature in range(len(features)):
            features[feature]=features[feature].to(device)
        label = data[-1].to(device)
        pre = model(features[0],features[1],"train")
        pre=pre.to(torch.float32)
        label=label.unsqueeze(-1).to(torch.float32)
        loss = criterion(pre, label)
        for i in range(pre.shape[0]):
            prelist.append(float(pre[i][0]))
            truelist.append(float(label[i]))
            f.write(str(float(label[i])))
            f.write('\t\t')
            f.write(str(float(pre[i][0])))
            f.write('\n')
        loss.backward()
        optimizer.step()
        epoch_loss += (loss.detach().item())
    epoch_loss /= (batch_id+1)
    return prelist,truelist,epoch_loss

def gcn_train(model,dataloader,optimizer,criterion,device,i,epoch,outDir,epsilon,alpha):
    model.train()
    model.to(device)
    f=open(f'./tmp/train/val{i}/train_'+str(epoch)+'.txt','w')
    prelist = []
    truelist = []
    epoch_loss=0.0
    for batch_id,data in enumerate(dataloader):
        optimizer.zero_grad()
        label = data.y
        #正常样本
        pre=model(data.x,data.edge_index,data.batch,data.edge_attr,False,device)
        # summary(model,(data.x,data.edge_index,data.batch,data.edge_attr))
        pre=pre.to(torch.float32)
        label=label.unsqueeze(-1).to(torch.float32).to(device)
        normal_loss = criterion(pre, label)

        #对抗样本
        grapth_adv=generate_adv_sample(data,epsilon)
        jsloss_adv = js_div(grapth_adv.x, data.x)#js散度 loss
        pre_adv = model(grapth_adv.x, grapth_adv.edge_index,grapth_adv.batch,grapth_adv.edge_attr,False,device)
        label_adv=grapth_adv.y
        label_adv=label_adv.unsqueeze(-1).to(torch.float32).to(device)
        mseloss_dva=F.mse_loss(pre_adv,label_adv)
        w=0.5
        loss_adv=w*jsloss_adv+(1-w)*mseloss_dva#对抗样本的总loss

        #总loss
        total_loss = alpha * loss_adv + (1 - alpha) * normal_loss
        for i in range(pre.shape[0]):
            prelist.append(float(pre[i][0]))
            truelist.append(float(label[i]))
            f.write(str(float(label[i])))
            f.write('\t\t')
            f.write(str(float(pre[i][0])))
            f.write('\n')
        total_loss.backward()
        optimizer.step()
        epoch_loss += (total_loss.detach().item())
    epoch_loss /= (batch_id+1)
    return prelist,truelist,epoch_loss
    