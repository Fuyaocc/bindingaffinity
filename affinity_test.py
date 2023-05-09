import os
import re
import torch
import math
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
from utils.readFoldX import readFoldXResult
from utils.getSASA import getDSSP
from models.affinity_net_gcn import AffinityNet
from utils.getInterfaceRate import getInterfaceRateAndSeq
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.metrics import mean_squared_error
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader,Data
from sklearn.manifold import TSNE

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

    for line in open(args.inputDir+'test_set.txt'):
        blocks=re.split('\t|\n',line)
        pdbname=blocks[0]
        complexdict[pdbname]=float(blocks[1])
    files=os.listdir("./data/graph/")
    graph_dict=set()
    for file in files:
        graph_dict.add(file.split("_")[0])
    
    resfeat=getAAOneHotPhys()
    
    featureList=[]
    labelList=[]
    i=0
    maxlen=0
    for pdbname in complexdict.keys():
        # if i>=1:
        #     continue
        # i=i+1
        #local redisue
        if pdbname.startswith("5ma"):continue #dssp can't cal rSA
        # if pdbname in graph_dict:
        #     logging.info("load graph :"+pdbname)
        #     x = torch.load('./data/graph/'+pdbname+"_x"+'.pth').to(args.device)
        #     #去掉onehot
        #     x = x[:,21:]
        #     edge_index=torch.load('./data/graph/'+pdbname+"edge_index"+'.pth').to(args.device)
        #     edge_attr=torch.load('./data/graph/'+pdbname+"_edge_attr"+'.pth').to(args.device)
        #     energy=torch.load('./data/graph/'+pdbname+"_energy"+'.pth').to(args.device)
        # else:
        if pdbname=="2jgz":
            plipinter=[]
            with open("./data/plip.txt",'r') as f:
                for line in f:
                    blocks=re.split("\t|\n",line)
                    plipinter.append(blocks[0])
                    plipinter.append(blocks[1])
            pdb_path='./data/pdbs/'+pdbname+'.pdb'
            energy=readFoldXResult(args.foldxPath,pdbname)
            energy=torch.tensor(energy,dtype=torch.float)
            seq,interfaceDict,chainlist,connect=getInterfaceRateAndSeq(pdb_path,plipinter,interfaceDis=args.interfacedis)
            print(interfaceDict)
            print(connect)
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
            # torch.save(x.to(torch.device('cpu')),'./data/graph/'+pdbname+"_x"+'.pth')
            # torch.save(edge_index.to(torch.device('cpu')),'./data/graph/'+pdbname+"edge_index"+'.pth')
            # torch.save(edge_attr.to(torch.device('cpu')),'./data/graph/'+pdbname+"_edge_attr"+'.pth')
            # torch.save(energy.to(torch.device('cpu')),'./data/graph/'+pdbname+"_energy"+'.pth')      
        
            data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr,y=complexdict[pdbname],name=pdbname,energy=energy)
            # if  data.num_nodes> 2500 : continue
            maxlen+=1
            featureList.append(data) 
            labelList.append(complexdict[pdbname])
        
    # for i in range(5):
    #     Affinity_model=AffinityNet(in_channels=args.dim
    #                                ,hidden_channels=args.dim//2
    #                                ,out_channels=args.dim//2
    #                                ,mlp_in_channels=args.dim//2
    #                                ,device=args.device)
    #     # Affinity_model.load_state_dict(torch.load(f"./models/saved/gcn/affinity_model{i}_dim{args.dim}.pt"))
    #     dataset=MyGCNDataset(featureList)
    #     dataloader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True)
    #     criterion = torch.nn.MSELoss()
        
    #     test_prelist, test_truelist,test_loss = gcn_predict(Affinity_model,dataloader,criterion,args.device,i,0)
    #     df = pd.DataFrame({'label':test_truelist, 'pre':test_prelist})
    #     mae=F.l1_loss(torch.tensor(test_prelist),torch.tensor(test_truelist))
    #     mse=F.mse_loss(torch.tensor(test_prelist),torch.tensor(test_truelist))
    #     rmse=math.sqrt(mse)
    #     test_pcc = df.pre.corr(df.label)
    #     print("Test Loss = %.4f"%(test_loss)+" , mse = %.4f"%(mse)+" , rmse = %.4f"%(rmse)+" , mae = %.4f"%(mae)+" , pcc = %.4f"%(test_pcc))
    #     with open(f'./tmp/pred/result_{i}.txt','w') as f:
    #         for j in range(0,len(test_truelist)):
    #             f.write(str(test_truelist[j]))
    #             f.write('\t\t')
    #             f.write(str(test_prelist[j]))
    #             f.write('\n')
            
        # logging.info(str(i)+" ,  MSE:"+str(mse)+" , rmse:"+str(rmse)+" , mae:"+str(mae)+" , PCC:"+str(test_pcc))
        