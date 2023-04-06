import os
import re
import torch
import numpy as np
import sys
import logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
import esm
import pandas as pd

from utils.parse_args import get_args
from utils.seq2ESM1v import seq2ESM1v
from utils.MyDataset import MyGCNDataset,gcn_pickfold
from utils.run_epoch import gcn_train
from utils.predict import gcn_predict
from utils.generateGraph import generate_residue_graph
from models.affinity_net_gcn import AffinityNet
from utils.getInterfaceRate import getInterfaceRateAndSeq
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.metrics import mean_squared_error
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader,Data

from do_test import check_len

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
    
    featureList=[]
    labelList=[]
    i=0
    maxlen=0
    for pdbname in complexdict.keys():
        # if i>=10:
        #     continue
        # i=i+1
        
        #local redisue
        seq,interfaceDict,clist,connect=getInterfaceRateAndSeq('../pdbs/'+pdbname+'.pdb',interfaceDis=args.interfacedis)
        cl1=interfaceDict[clist[0]]
        cl2=interfaceDict[clist[1]]
        start1=seq[pdbname+'_'+clist[0]][1]
        start2=seq[pdbname+'_'+clist[1]][1]

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
        node_feature={}
        for v in cl1:
            reduise=v.split('_')[1]
            index=int(reduise[1:])-start1
            node_feature[v]=[]
            node_feature[v].append(c1_all[index].tolist())
            node_feature[v].append(esm_c1_all[index].tolist())
        for v in cl2:
            reduise=v.split('_')[1]
            index=int(reduise[1:])-start2
            node_feature[v]=[]
            node_feature[v].append(c2_all[index].tolist())
            node_feature[v].append(esm_c2_all[index].tolist())
            
        node_features, adj=generate_residue_graph(pdbname,node_feature,connect,args.padding)
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index,y=complexdict[pdbname])
        
        featureList.append(data) 
        labelList.append(complexdict[pdbname])
        
    #10折交叉
    kf = KFold(n_splits=10,random_state=2022, shuffle=True)
    best_pcc=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    best_mse=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    best_epoch=[0,0,0,0,0,0,0,0,0,0]
    for i, (train_index, test_index) in enumerate(kf.split(np.array(labelList))):
        
        Affinity_model=AffinityNet(num_features=args.dim,hidden_channel=args.dim//2,out_channel=(args.dim//2)//2,device=args.device)

        train_set,val_set=gcn_pickfold(featureList,train_index,test_index)
        
        train_dataset=MyGCNDataset(train_set)
        train_dataloader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)

        test_dataset=MyGCNDataset(val_set)
        test_dataloader=DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(Affinity_model.parameters(), lr = 1e-4, weight_decay = 1e-4)

        writer = SummaryWriter('./log/val'+str(i))
        
        for epoch in range(args.epoch):
            train_prelist, train_truelist, train_loss = gcn_train(Affinity_model,train_dataloader,optimizer,criterion,args.device,i,epoch,args.outDir,args.num_layers)
            logging.info("Epoch "+ str(epoch)+ ": train Loss = %.4f"%(train_loss))

            df = pd.DataFrame({'label':train_truelist, 'pre':train_prelist})
            train_pcc = df.pre.corr(df.label)
            writer.add_scalar('affinity_train/loss', train_loss, epoch)
            writer.add_scalar('affinity_train/pcc', train_pcc, epoch)
        
            test_prelist, test_truelist,test_loss = gcn_predict(Affinity_model,test_dataloader,criterion,args.device,i,epoch,args.num_layers)
            logging.info("Epoch "+ str(epoch)+ ": test Loss = %.4f"%(test_loss))

            df = pd.DataFrame({'label':test_truelist, 'pre':test_prelist})
            test_pcc = df.pre.corr(df.label)
            writer.add_scalar('affinity_test/loss', test_loss, epoch)
            writer.add_scalar('affinity_test/pcc', test_pcc, epoch)
            
            if test_pcc > best_pcc[i]:
                best_pcc[i]=test_pcc
                best_mse[i]=mean_squared_error(test_truelist,test_prelist)
                best_epoch[i]=epoch
                torch.save(Affinity_model,f'./models/saved/affinitymodel_{i}.pt')
    
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
            