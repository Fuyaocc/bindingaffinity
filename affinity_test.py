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
    
    resFeature=getAAOneHotPhys()
    
    featureList=[]
    labelList=[]
    i=0
    maxlen=0
    for pdbname in complexdict.keys():
        # if i>=30:
        #     continue
        # i=i+1

        #local redisue
        seq,interfaceDict,chainlist,connect=getInterfaceRateAndSeq('./data/pdbs/'+pdbname+'.pdb',interfaceDis=args.interfacedis)
        #esm1v seq embedding
        # logging.info("generate sequcence esm1v : "+pdbname)
        # chain1_esm=torch.load(args.inputDir+'esmfeature/'+pdbname+'_chain1.pth')
        # chain2_esm=torch.load(args.inputDir+'esmfeature/'+pdbname+'_chain2.pth')
        # seqfeat.append(chain1_esm.to(args.device))
        # seqfeat.append(chain2_esm.to(args.device))
        
        #esm-if1 struct embedding
        # logging.info("loading esm structure emb :  "+pdbname)
        # esm_c1_all=torch.load('./data/esmfeature/strute_emb/'+pdbname+'_'+chainlist[0]+'.pth')
        # esm_c2_all=torch.load('./data/esmfeature/strute_emb/'+pdbname+'_'+chainlist[1]+'.pth')
        # esm_c1_all=F.avg_pool1d(esm_c1_all,16,16)
        # esm_c2_all=F.avg_pool1d(esm_c2_all,16,16)
        
        #le embedding
        # logging.info("loading prodesign le emb :  "+pdbname)
        # c1_all=torch.load('./data/lefeature/'+pdbname+'_'+clist[0]+'.pth')
        # c2_all=torch.load('./data/lefeature/'+pdbname+'_'+clist[1]+'.pth')
        # c1_all=F.max_pool1d(c1_all,16,16)
        # c2_all=F.max_pool1d(c2_all,16,16)
        
        node_feature={}
        logging.info("generate graph :"+pdbname)
        for chain in chainlist:
            reslist=interfaceDict[chain]
            esm1f_feat=torch.load('./data/esmfeature/strute_emb/'+pdbname+'_'+chain+'.pth')
            esm1f_feat=F.avg_pool1d(esm1f_feat,16,16)
            for v in reslist:
                reduise=v.split('_')[1]
                index=int(reduise[1:])-1
                node_feature[v]=[]
                # node_feature[v].append(c1_all[index].tolist())
                node_feature[v].append(esm1f_feat[index].tolist())
                node_feature[v].append(resFeature[reduise[0]])
            
        node_features, edge_index,edge_attr=generate_residue_graph(pdbname,node_feature,connect,args.padding)
        if(len(node_feature)==0):continue
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index=torch.tensor(edge_index,dtype=torch.long).t().contiguous()
        edge_attr=torch.tensor(edge_attr,dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr,y=complexdict[pdbname])
                
        featureList.append(data) 
        labelList.append(complexdict[pdbname])
        
    
    Affinity_model=AffinityNet(num_features=args.dim,hidden_channel=args.dim//2,out_channel=args.dim//2,device=args.device)
    Affinity_model.load_state_dict(torch.load("./models/saved/gcn/affinity_model2.pt"))
    dataset=MyGCNDataset(featureList)
    dataloader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True)
    criterion = torch.nn.MSELoss()
    
    test_prelist, test_truelist,test_loss = gcn_predict(Affinity_model,dataloader,criterion,args.device,i,0,args.num_layers)
    logging.info(" Loss = %.4f"%(test_loss))
    df = pd.DataFrame({'label':test_truelist, 'pre':test_prelist})
    with open(f'./tmp/pred.txt','w') as f:
        for i in range(0,len(test_truelist)):
            f.write(str(test_truelist[i]))
            f.write('\t\t')
            f.write(str(test_prelist[i]))
            f.write('\n')
        
    test_pcc = df.pre.corr(df.label)
    logging.info("pcc:"+str(test_pcc))
        