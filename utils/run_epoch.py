import torch
import logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
# from  torchsummary import summary

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

def gcn_train(model,dataloader,optimizer,criterion,device,i,epoch,outDir,num_layers):
    model.train()
    model.to(device)
    f=open(f'./tmp/train/val{i}/train_'+str(epoch)+'.txt','w')
    prelist = []
    truelist = []
    epoch_loss=0.0
    for batch_id,data in enumerate(dataloader):
        optimizer.zero_grad()
        label = data.y
        pre=model(data.x,data.edge_index,data.batch,data.edge_attr,False,device)
        # summary(model,(data.x,data.edge_index,data.batch,data.edge_attr))
        pre=pre.to(torch.float32)
        label=label.unsqueeze(-1).to(torch.float32).to(device)
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
    