import os
import re
import torch
import math
import numpy as np
import sys
import logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
import pandas as pd
import torch.nn.functional as F
import torch.optim.lr_scheduler as ls

from utils.parse_args import get_args
from utils.seq2ESM1v import seq2ESM1v
from utils.MyDataset import MyGCNDataset,gcn_pickfold
from utils.run_epoch import mpnn_train,gcn_train
from utils.predict import gcn_predict
from utils.generateGraph import generate_residue_graph
from utils.resFeature import getAAOneHotPhys,getSignPhys
from utils.getSASA import getDSSP
from utils.readFoldX import readFoldXResult
from models.affinity_net_mpnn import Net
from utils.getInterfaceRate import getInterfaceRateAndSeq
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.metrics import mean_squared_error
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader,Data,Batch

def collate_fn(data_list):
    return Batch.from_data_list(data_list)

if __name__ == '__main__':
    args=get_args()
    print(args)

    complexdict={} # pdbname : [seq1, seq2, bingding_affinity]

    for line in open(args.inputDir+'pdbbind_data.txt'):
        blocks=re.split('\t|\n',line)
        pdbname=blocks[0]
        complexdict[pdbname]=float(blocks[1])
        
    filter_set=set(["1de4","1ogy","1tzn","2wss","3lk4","3sua","3swp","4r8i","6c74"])#data in test_set or dirty_data
    for line in open(args.inputDir+'test_set.txt'):
        blocks=re.split('\t|\n',line)
        filter_set.add(blocks[0])
    
    files=os.listdir("./data/graph/")
    graph_dict=set()
    for file in files:
        graph_dict.add(file.split("_")[0])
        
    resfeat=getAAOneHotPhys()

    featureList=[]
    labelList=[]
    i=0
    maxlen=0
    plipinter=[]
    for pdbname in complexdict.keys():
        # if i>= 32:
        #     continue
        # i=i+1
        # if pdbname != "2vlp":continue
        if pdbname in filter_set:continue
        #local redisue
        if pdbname.startswith("5ma"):continue #dssp can't cal rSA
        if pdbname in graph_dict:
            logging.info("load graph :"+pdbname)
            x = torch.load('./data/graph/'+pdbname+"_x"+'.pth').to(args.device)
            edge_index=torch.load('./data/graph/'+pdbname+"edge_index"+'.pth').to(args.device)
            edge_attr=torch.load('./data/graph/'+pdbname+"_edge_attr"+'.pth').to(args.device)
            energy=torch.load('./data/graph/'+pdbname+"_energy"+'.pth').to(args.device)
            # if(x.shape[0]>1000):continue
            # paddinglen=1000-x.shape[0]
            # padding=torch.zeros(paddinglen,x.shape[1]).to(args.device)
            # x = torch.cat([x,padding],dim=0)


        else:
            pdb_path='./data/pdbs/'+pdbname+'.pdb'
            energy=readFoldXResult(args.foldxPath,pdbname)
            energy=torch.tensor(energy,dtype=torch.float)
            seq,interfaceDict,chainlist,connect=getInterfaceRateAndSeq(pdb_path,plipinter,interfaceDis=args.interfacedis)
            dssp=getDSSP(pdb_path)
            node_feature={}
            logging.info("generate graph :"+pdbname)
            for chain in chainlist:
                reslist=interfaceDict[chain]
                esm1f_feat=torch.load('./data/esmfeature/strute_emb/'+pdbname+'_'+chain+'.pth')
                esm1f_feat=F.avg_pool1d(esm1f_feat,16,16)
                # le_feat=torch.load('./data/lefeature/'+pdbname+'_'+chain+'.pth')
                # le_feat=F.avg_pool1d(le_feat,16,16)
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
            torch.save(x.to(torch.device('cpu')),'./data/graph/'+pdbname+"_x"+'.pth')
            torch.save(edge_index.to(torch.device('cpu')),'./data/graph/'+pdbname+"edge_index"+'.pth')
            torch.save(edge_attr.to(torch.device('cpu')),'./data/graph/'+pdbname+"_edge_attr"+'.pth')
            torch.save(energy.to(torch.device('cpu')),'./data/graph/'+pdbname+"_energy"+'.pth')      
        data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr,y=complexdict[pdbname],name=pdbname,energy=energy)

        maxlen+=1
        featureList.append(data) 
        labelList.append(complexdict[pdbname])
    logging.info(maxlen)
    #交叉验证
    kf = KFold(n_splits=5,random_state=2022, shuffle=True)
    best_pcc=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    best_mse=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    best_rmse=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    best_mae=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    best_epoch=[0,0,0,0,0,0,0,0,0,0]
    for i, (train_index, test_index) in enumerate(kf.split(np.array(labelList))):
        net=Net(input_dim=args.dim
                        ,hidden_dim=args.dim
                        ,output_dim=args.dim)
        
        net.to(args.device)

        train_set,val_set=gcn_pickfold(featureList,train_index,test_index)
        
        train_dataset=MyGCNDataset(train_set)
        train_dataloader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True, collate_fn=collate_fn)

        val_dataset=MyGCNDataset(val_set)
        val_dataloader=DataLoader(val_dataset,batch_size=args.batch_size,shuffle=True, collate_fn=collate_fn)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr = 1e-3, weight_decay = 1e-3)

        writer = SummaryWriter('./log/val'+str(i))
        
        for epoch in range(args.epoch):
            #train
            net,train_prelist, train_truelist, train_loss = mpnn_train(net
                                                                                                        ,train_dataloader
                                                                                                        ,optimizer
                                                                                                        ,criterion
                                                                                                        ,args.device
                                                                                                        ,i
                                                                                                        ,epoch
                                                                                                        ,args.outDir
                                                                                                        ,args.epsilon
                                                                                                        ,args.alpha)
                                                                                                   
            logging.info("Epoch "+ str(epoch)+ ": train Loss = %.4f"%(train_loss))

            df = pd.DataFrame({'label':train_truelist, 'pre':train_prelist})
            train_pcc = df.pre.corr(df.label)
            writer.add_scalar('affinity_train/loss', train_loss, epoch)
            writer.add_scalar('affinity_train/pcc', train_pcc, epoch)
            
            #val
            val_prelist, val_truelist,val_loss = gcn_predict(net,val_dataloader,criterion,args.device,i,epoch)
            df = pd.DataFrame({'label':val_truelist, 'pre':val_prelist})
            val_pcc = df.pre.corr(df.label)
            writer.add_scalar('affinity_val/loss', val_loss, epoch)
            writer.add_scalar('affinity_val/pcc', val_pcc, epoch)
            mae=F.l1_loss(torch.tensor(val_prelist),torch.tensor(val_truelist))
            mse=F.mse_loss(torch.tensor(val_prelist),torch.tensor(val_truelist))
            rmse=math.sqrt(mse)
            logging.info("Epoch "+ str(epoch)+ ": val Loss = %.4f"%(val_loss)+" , mse = %.4f"%(mse)+" , rmse = %.4f"%(rmse)+" , mae = %.4f"%(mae))
            if rmse < 2.78 and val_pcc > best_pcc[i]:
                best_pcc[i]=val_pcc
                best_mse[i]=mse
                best_rmse[i]=rmse
                best_mae[i]=mae
                best_epoch[i]=epoch
                torch.save(net.state_dict(),f'./models/saved/gcn/affinity_model{i}_dim{args.dim}_foldx.pt')
    
    pcc=0.
    mse=0.
    rmse=0.
    mae=0.
    for i in range(5):
        pcc=pcc+best_pcc[i]
        mse=mse+best_mse[i]
        rmse=rmse+best_rmse[i]
        mae=mae+best_mae[i]
        logging.info('val_'+str(i)+' best_pcc = %.4f'%(best_pcc[i])+' , best_mse = %.4f'%(best_mse[i])+' , best_rmse = %.4f'%(best_rmse[i])+' , best_mae = %.4f'%(best_mae[i])+' , best_epoch : '+str(best_epoch[i]))

    
    print('pcc  :   '+str(pcc/5))
    print('mse  :   '+str(mse/5))
    print('rmse :   '+str(rmse/5))
    print('mae  :   '+str(mae/5))
            