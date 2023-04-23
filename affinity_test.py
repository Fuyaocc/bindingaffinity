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
    
    resfeat=getAAOneHotPhys()
    
    featureList=[]
    labelList=[]
    i=0
    maxlen=0
    for pdbname in complexdict.keys():
        # if i>=30:
        #     continue
        # i=i+1

        #local redisue
        pdb_path='./data/pdbs/'+pdbname+'.pdb'
        if pdbname.startswith("5ma"):continue
        seq,interfaceDict,chainlist,connect=getInterfaceRateAndSeq('./data/pdbs/'+pdbname+'.pdb',interfaceDis=args.interfacedis)
        #esm1v seq embedding
        # logging.info("generate sequcence esm1v : "+pdbname)
        # chain1_esm=torch.load(args.inputDir+'esmfeature/'+pdbname+'_chain1.pth')
        # chain2_esm=torch.load(args.inputDir+'esmfeature/'+pdbname+'_chain2.pth')
        # seqfeat.append(chain1_esm.to(args.device))
        # seqfeat.append(chain2_esm.to(args.device))

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
        
    for i in range(10):
        Affinity_model=AffinityNet(num_features=args.dim,hidden_channel=args.dim//2,out_channel=args.dim//2,device=args.device)
        Affinity_model.load_state_dict(torch.load(f"./models/saved/gcn/affinity_model{i}.pt"))
        dataset=MyGCNDataset(featureList)
        dataloader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True)
        criterion = torch.nn.MSELoss()
        
        test_prelist, test_truelist,test_loss = gcn_predict(Affinity_model,dataloader,criterion,args.device,i,0)
        df = pd.DataFrame({'label':test_truelist, 'pre':test_prelist})
        with open(f'./tmp/pred/result_{i}.txt','w') as f:
            for j in range(0,len(test_truelist)):
                f.write(str(test_truelist[j]))
                f.write('\t\t')
                f.write(str(test_prelist[j]))
                f.write('\n')
            
        test_pcc = df.pre.corr(df.label)
        logging.info(str(i)+" ,  MSE:"+str(test_loss)+" , PCC:"+str(test_pcc))
        