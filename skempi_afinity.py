import os
import re
import torch
import numpy as np
import sys
import logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
import esm
import pandas as pd

from torch.utils.data import DataLoader,Dataset
# sys.path.append('./utils')
from utils.parse_args import get_args
from utils.seq2ESM1v import seq2ESM1v
from utils.MyDataset import MyDataset,pickfold
from utils.run_epoch import run_train
from utils.predict import run_predict
from models.affinity_net import AffinityNet
from sklearn.model_selection import KFold, ShuffleSplit
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    args=get_args()
    print(args)

    complexdict={} # pdbname : [seq1, seq2, bingding_affinity]
    seqdict={}
    for line in open(args.inputDir+'SKEMPI_seq.txt'):
        blocks=re.split('\t|\n',line)
        seqdict[blocks[0]]=blocks[1]

    for line in open(args.inputDir+'SKEMPI_all_dg_avg.txt'):
        blocks=re.split('\t|\n',line)
        pdbname=blocks[0]+'+'+blocks[1]
        data=[]
        data.append(seqdict[blocks[0]])
        data.append(seqdict[blocks[1]])
        data.append(float(blocks[2]))
        complexdict[pdbname]=data
    
    # print(complexdict)

    featureList=[]
    labelList=[]
    i=0
    logging.info("loading sequcence esm1v  feature")
    for pdbname in complexdict.keys():
        # if i>=10:
        #     continue
        # i=i+1
        #structe 
        #to do
        #sequence
        esmFeature=[]
        chain_name=pdbname.split('+')
        chain1_esm=torch.load(args.inputDir+'esmfeature/skempi/'+chain_name[0]+'.pth')
        chain2_esm=torch.load(args.inputDir+'esmfeature/skempi/'+chain_name[1]+'.pth')
        esmFeature.append(chain1_esm)
        esmFeature.append(chain2_esm)
        featureList.append( esmFeature)

        labelList.append(complexdict[pdbname][2])
    logging.info("loading sequcence esm1v  feature  finish")
    for i in range(len(featureList)):
        for j in range(2):
            # print(featureList[i][j].shape)
            paddinglen=0
            if featureList[i][j].size(0)<=1000:
                paddinglen=1000-featureList[i][j].size(0)
                padding=torch.zeros(paddinglen,1280)
                featureList[i][j]=torch.cat((featureList[i][j],padding),dim=0)
        

    #10折交叉
    kf = KFold(n_splits=10,random_state=2022, shuffle=True)
    best_pcc=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    best_mse=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    for i, (train_index, test_index) in enumerate(kf.split(np.array(labelList))):
        Affinity_model=AffinityNet(dim=args.dim,device=args.device)

        x_train,y_train,x_test,y_test=pickfold(featureList,labelList,train_index,test_index)
        train_dataset=MyDataset(x_train,y_train)
        train_dataloader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)

        test_dataset=MyDataset(x_test,y_test)
        test_dataloader=DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(Affinity_model.parameters(), lr = 1e-4, weight_decay = 1e-3)

        writer = SummaryWriter('./log/skempi/val'+str(i))
        
        for epoch in range(args.epoch):
            train_prelist, train_truelist, train_loss = run_train(Affinity_model,train_dataloader,optimizer,criterion,args.device,epoch,args.outDir,args.num_layers)
            logging.info("Epoch "+ str(epoch)+ ": train Loss = %.4f"%(train_loss))

            df = pd.DataFrame({'label':train_truelist, 'pre':train_prelist})
            train_pcc = df.pre.corr(df.label)
            writer.add_scalar('affinity_train/loss', train_loss, epoch)
            writer.add_scalar('affinity_train/pcc', train_pcc, epoch)
        
            test_prelist, test_truelist,test_loss = run_predict(Affinity_model,test_dataloader,criterion,args.device,args.num_layers)
            logging.info("Epoch "+ str(epoch)+ ": test Loss = %.4f"%(test_loss))

            df = pd.DataFrame({'label':test_truelist, 'pre':test_prelist})
            test_pcc = df.pre.corr(df.label)
            writer.add_scalar('affinity_test/loss', test_loss, epoch)
            writer.add_scalar('affinity_test/pcc', test_pcc, epoch)
            
            if test_pcc > best_pcc[i]:
                best_pcc[i]=test_pcc
                best_mse[i]=test_loss
                torch.save(Affinity_model,f'./models/saved/skempi/skempi_affinitymodel_{i}.pt')
    
    pcc=0
    mse=0
    for i in range(10):
        pcc=pcc+best_pcc[i]
        mse=mse+best_mse[i]
        print(f'val_{i}   best_pcc  :   '+str(best_pcc[i]))
        print(f'val_{i}   best_mse  :   '+str(best_mse[i]))
    
    print('pcc  :   '+str(pcc/10))
    print('mse  :   '+str(mse/10))
            
            
        
