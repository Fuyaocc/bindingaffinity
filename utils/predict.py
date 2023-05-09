import torch
import logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

def run_predict(model,dataloader,criterion,device,i,epoch,num_layers):
    model.eval()
    model.to(device)
    epoch_loss=0
    dataitr=iter(dataloader)
    f=open(f'./tmp/test/val{i}/test_'+str(epoch)+'.txt','w')
    prelist = []
    truelist = []
    for batch_id,data in enumerate(dataitr):
        features=data[:-1]
        for feature in range(len(features)):
            features[feature]=features[feature].to(device)
        label = data[-1].to(device)
        pre = model(features[0],features[1],num_layers,False)
        pre=pre.to(torch.float32)
        label=label.unsqueeze(-1).to(torch.float32)
        loss = criterion(pre, label)
        for i in range(pre.shape[0]):
            prelist.append(float(pre[i][0]))
            truelist.append(float(label[i][0]))
            f.write(str(float(label[i])))
            f.write('\t\t')
            f.write(str(float(pre[i][0])))
            f.write('\n')
        epoch_loss += (loss.detach().item())
    epoch_loss /= (batch_id+1)
    return prelist,truelist,epoch_loss


def gcn_predict(model,dataloader,criterion,device,i,epoch):
    model.eval()
    model.to(device)
    epoch_loss=0
    f=open(f'./tmp/test/val{i}/test_'+str(epoch)+'.txt','w')
    prelist = []
    truelist = []
    for batch_id,data in enumerate(dataloader):
        label = data.y
        names=data.name
        pre=model(data,False,device)
        pre=pre.to(torch.float32)
        label=label.unsqueeze(-1).to(torch.float32).to(device)
        loss = criterion(pre, label)
        for i in range(pre.shape[0]):
            prelist.append(float(pre[i][0]))
            truelist.append(float(label[i][0]))
            f.write(names[i])
            f.write('\t\t')
            f.write(str(float(label[i][0])))
            f.write('\t\t')
            f.write(str(float(pre[i][0])))
            f.write('\n')
        epoch_loss += (loss.detach().item())
    epoch_loss /= (batch_id+1)
    return prelist,truelist,epoch_loss

