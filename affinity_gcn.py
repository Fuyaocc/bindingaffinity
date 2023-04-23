import os
import re
import torch
import numpy as np
import sys
import logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
import esm
import pandas as pd
import torch.nn.functional as F

from utils.parse_args import get_args
from utils.seq2ESM1v import seq2ESM1v
from utils.MyDataset import MyGCNDataset,gcn_pickfold
from utils.run_epoch import gcn_train
from utils.predict import gcn_predict
from utils.generateGraph import generate_residue_graph
from utils.resFeature import getAAOneHotPhys
from utils.getSASA import getDSSP
from models.affinity_net_gcn import AffinityNet
from utils.getInterfaceRate import getInterfaceRateAndSeq
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.metrics import mean_squared_error
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader,Data


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
    # for line in open(args.inputDir+'test_set.txt'):
    #     blocks=re.split('\t|\n',line)
    #     pdbname=blocks[0]
    #     complexdict[pdbname]=float(blocks[1])
        
    resfeat=getAAOneHotPhys()
    
    # bad_case={"4r8i"}

    featureList=[]
    labelList=[]
    i=0
    maxlen=0
    for pdbname in complexdict.keys():
        # if i>=1:
        #     continue
        # i=i+1
        # if pdbname != "2vlp":continue

        #local redisue
        pdb_path='./data/pdbs/'+pdbname+'.pdb'
        if pdbname.startswith("5ma"):continue
        seq,interfaceDict,chainlist,connect=getInterfaceRateAndSeq(pdb_path,interfaceDis=args.interfacedis)
        #esm1v seq embedding
        # logging.info("generate sequcence esm1v : "+pdbname)
        # chain1_esm=torch.load(args.inputDir+'esmfeature/'+pdbname+'_chain1.pth')
        # chain2_esm=torch.load(args.inputDir+'esmfeature/'+pdbname+'_chain2.pth')
        # seqfeat.append(chain1_esm.to(args.device))
        # seqfeat.append(chain2_esm.to(args.device))
        
        #le embedding
        # logging.info("loading prodesign le emb :  "+pdbname)
        # c1_all=torch.load('./data/lefeature/'+pdbname+'_'+clist[0]+'.pth')
        # c2_all=torch.load('./data/lefeature/'+pdbname+'_'+clist[1]+'.pth')
        # c1_all=F.max_pool1d(c1_all,16,16)
        # c2_all=F.max_pool1d(c2_all,16,16)

        dssp=getDSSP(pdb_path)
        node_feature={}
        logging.info("generate graph :"+pdbname)
        for chain in chainlist:
            reslist=interfaceDict[chain]
            esm1f_feat=torch.load('./data/esmfeature/strute_emb/'+pdbname+'_'+chain+'.pth')
            esm1f_feat=F.avg_pool1d(esm1f_feat,16,16)
            le_feat=torch.load('./data/lefeature/'+pdbname+'_'+chain+'.pth')
            le_feat=F.avg_pool1d(le_feat,16,16)
            for v in reslist:
                reduise=v.split('_')[1]
                index=int(reduise[1:])-1
                if (chain,index+1) not in dssp.keys(): 
                    other_feat=[0.0]
                else:          
                    other_feat=[dssp[(chain,index+1)][3]]#[rSA,...]
                node_feature[v]=[]
                node_feature[v].append(resfeat[reduise[0]])
                node_feature[v].append(other_feat)
                # node_feature[v].append(le_feat[index].tolist())
                node_feature[v].append(esm1f_feat[index].tolist())
                
            
        node_features, edge_index,edge_attr=generate_residue_graph(pdbname,node_feature,connect,args.padding)
        if(len(node_feature)==0):continue
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index=torch.tensor(edge_index,dtype=torch.long).t().contiguous()
        edge_attr=torch.tensor(edge_attr,dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr,y=complexdict[pdbname])
                
        featureList.append(data) 
        labelList.append(complexdict[pdbname])

    #10折交叉
    kf = KFold(n_splits=10,random_state=2022, shuffle=True)
    best_pcc=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    best_mse=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    best_epoch=[0,0,0,0,0,0,0,0,0,0]
    for i, (train_index, test_index) in enumerate(kf.split(np.array(labelList))):
        
        Affinity_model=AffinityNet(num_features=args.dim,hidden_channel=args.dim//2,out_channel=args.dim//2,device=args.device)

        train_set,val_set=gcn_pickfold(featureList,train_index,test_index)
        
        train_dataset=MyGCNDataset(train_set)
        train_dataloader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,drop_last=True)

        val_dataset=MyGCNDataset(val_set)
        val_dataloader=DataLoader(val_dataset,batch_size=args.batch_size,shuffle=True)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(Affinity_model.parameters(), lr = 1e-3, weight_decay = 1e-4)

        writer = SummaryWriter('./log/val'+str(i))
        
        for epoch in range(args.epoch):
            train_prelist, train_truelist, train_loss = gcn_train(Affinity_model,train_dataloader,optimizer,criterion,args.device,i,epoch,args.outDir,args.epsilon,args.alpha)
            logging.info("Epoch "+ str(epoch)+ ": train Loss = %.4f"%(train_loss))

            df = pd.DataFrame({'label':train_truelist, 'pre':train_prelist})
            train_pcc = df.pre.corr(df.label)
            writer.add_scalar('affinity_train/loss', train_loss, epoch)
            writer.add_scalar('affinity_train/pcc', train_pcc, epoch)
        
            val_prelist, val_truelist,val_loss = gcn_predict(Affinity_model,val_dataloader,criterion,args.device,i,epoch)
            logging.info("Epoch "+ str(epoch)+ ": test Loss = %.4f"%(val_loss))

            df = pd.DataFrame({'label':val_truelist, 'pre':val_prelist})
            val_pcc = df.pre.corr(df.label)
            writer.add_scalar('affinity_val/loss', val_loss, epoch)
            writer.add_scalar('affinity_val/pcc', val_pcc, epoch)
            
            if val_pcc > best_pcc[i]:
                best_pcc[i]=val_pcc
                best_mse[i]=mean_squared_error(val_truelist,val_prelist)
                best_epoch[i]=epoch
                torch.save(Affinity_model.state_dict(),f'./models/saved/gcn/affinity_model{i}.pt')
    
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
            