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
from utils.feature_padding import to_padding
from utils.generateGraph import generate_residue_graph
from models.affinity_net_v2 import AffinityNet
from utils.getInterfaceRate import getInterfaceRateAndSeq
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.metrics import mean_squared_error
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    args=get_args()
    print(args)

    complexdict={} # pdbname : [seq1, seq2, bingding_affinity]
    # seqdict={}
    # for line in open(args.inputDir+'pdbbind_seq.txt'):
    #     blocks=re.split('\t|\n',line)
    #     pdbname=blocks[0].split('_')[0]
    #     if pdbname in seqdict.keys():
    #         seqdict[pdbname].append(blocks[1])
    #     else:
    #         seqs=[]
    #         seqs.append(blocks[1])
    #         seqdict[pdbname]=seqs

    for line in open(args.inputDir+'train_set.txt'):
        blocks=re.split('\t|\n',line)
        pdbname=blocks[0]
        complexdict[pdbname]=float(blocks[1])
    
    # print(complexdict)

    featureList=[]
    labelList=[]
    i=0
    maxlen=0
    for pdbname in complexdict.keys():
        if i>=30:
            continue
        i=i+1
                
        #local redisue
        c1=[]
        c2=[]
        seq,interfaceDict,clist,connect=getInterfaceRateAndSeq('./data/pdbs/'+pdbname+'.pdb',interfaceDis=args.interfacedis)
        cl1=interfaceDict[clist[0]]
        cl2=interfaceDict[clist[1]]
        start1=seq[pdbname+'_'+clist[0]][1]
        start2=seq[pdbname+'_'+clist[1]][1]

        esmFeature=[]
        #esm1v seq embedding
        # logging.info("generate sequcence esm1v : "+pdbname)
        # chain1_esm=torch.load(args.inputDir+'esmfeature/'+pdbname+'_chain1.pth')
        # chain2_esm=torch.load(args.inputDir+'esmfeature/'+pdbname+'_chain2.pth')
        # esmFeature.append(chain1_esm.to(args.device))
        # esmFeature.append(chain2_esm.to(args.device))
        
        #esm-if1 struct embedding
        logging.info("loading esm structure emb :  "+pdbname)
        esm_c1_all=torch.load('./data/esmfeature/strute_emb/'+pdbname+'_'+clist[0]+'.pth')
        esm_c2_all=torch.load('./data/esmfeature/strute_emb/'+pdbname+'_'+clist[1]+'.pth')
        
        #le embedding
        logging.info("loading prodesign le emb :  "+pdbname)
        c1_all=torch.load('./data/lefeature/'+pdbname+'_'+clist[0]+'.pth')
        c2_all=torch.load('./data/lefeature/'+pdbname+'_'+clist[1]+'.pth')
        
        for v in cl1:
            reduise=v.split('_')[1]
            index=int(reduise[1:])-start1
            c1.append(c1_all[index].tolist()+esm_c1_all[index].tolist())
            
        for v in cl2:
            reduise=v.split('_')[1]
            index=int(reduise[1:])-start2
            c2.append(c2_all[index].tolist()+esm_c2_all[index].tolist())
                   
        
        if maxlen < len(c1): maxlen= len(c1)
        if maxlen < len(c2): maxlen= len(c2)
        
        esmFeature.append(torch.Tensor(c1).to(args.device))
        esmFeature.append(torch.Tensor(c2).to(args.device))
        esmFeature.append(pdbname)
        
        featureList.append( esmFeature) 
        labelList.append(complexdict[pdbname])
    
    logging.info(str(maxlen))
    
    #padding
    featureList=to_padding(featureList)
        
    #10折交叉
    kf = KFold(n_splits=10,random_state=2022, shuffle=True)
    best_pcc=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    best_mse=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    best_epoch=[0,0,0,0,0,0,0,0,0,0]
    for i, (train_index, test_index) in enumerate(kf.split(np.array(labelList))):
        # if i!=6:continue
        Affinity_model=AffinityNet(dim=args.dim,len=args.padding,device=args.device)

        x_train,y_train,x_test,y_test=pickfold(featureList,labelList,train_index,test_index)
        
        # check_len(i,x_train,x_test)
        
        train_dataset=MyDataset(x_train,y_train)
        train_dataloader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)

        test_dataset=MyDataset(x_test,y_test)
        test_dataloader=DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(Affinity_model.parameters(), lr = 1e-4, weight_decay = 1e-4)
        # optimizer = torch.optim.AdamW(Affinity_model.parameters(),lr=1e-4,weight_decay=1e-4)

        writer = SummaryWriter('./log/val'+str(i))
        
        for epoch in range(args.epoch):
            train_prelist, train_truelist, train_loss = run_train(Affinity_model,train_dataloader,optimizer,criterion,args.device,i,epoch,args.outDir,args.num_layers)
            logging.info("Epoch "+ str(epoch)+ ": train Loss = %.4f"%(train_loss))

            df = pd.DataFrame({'label':train_truelist, 'pre':train_prelist})
            train_pcc = df.pre.corr(df.label)
            writer.add_scalar('affinity_train/loss', train_loss, epoch)
            writer.add_scalar('affinity_train/pcc', train_pcc, epoch)
        
            test_prelist, test_truelist,test_loss = run_predict(Affinity_model,test_dataloader,criterion,args.device,i,epoch,args.num_layers)
            logging.info("Epoch "+ str(epoch)+ ": test Loss = %.4f"%(test_loss))

            df = pd.DataFrame({'label':test_truelist, 'pre':test_prelist})
            test_pcc = df.pre.corr(df.label)
            writer.add_scalar('affinity_test/loss', test_loss, epoch)
            writer.add_scalar('affinity_test/pcc', test_pcc, epoch)
            
            if test_pcc > best_pcc[i]:
                best_pcc[i]=test_pcc
                best_mse[i]=mean_squared_error(test_truelist,test_prelist)
                best_epoch[i]=epoch
                torch.save(Affinity_model,f'./models/saved/cnn_att/affinitymodel_{i}.pt')
    
    pcc=0
    mse=0
    for i in range(10):
        pcc=pcc+best_pcc[i]
        mse=mse+best_mse[i]
        print(f'val_{i}   best_pcc  :   '+str(best_pcc[i]))
        print(f'val_{i}   best_mse  :   '+str(best_mse[i]))
        print(f'val_{i}   best_epoch:   '+str(best_epoch[i]))
    
    print('pcc  :   '+str(pcc/10))
    print('mse  :   '+str(mse/10))
            