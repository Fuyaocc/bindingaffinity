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
    # tsne = TSNE(n_components=32,method='exact')
    
    resFeature=getAAOneHotPhys()
    
    featureList=[]
    labelList=[]
    i=0
    maxlen=0
    for pdbname in complexdict.keys():
        # if i>=10:
        #     continue
        # i=i+1
        
        #local redisue
        seq,interfaceDict,clist,connect=getInterfaceRateAndSeq('./data/pdbs/'+pdbname+'.pdb',interfaceDis=args.interfacedis)
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
        esm_c1_all=F.max_pool1d(esm_c1_all,32,32)
        esm_c2_all=F.max_pool1d(esm_c2_all,32,32)
        
        #le embedding
        logging.info("loading prodesign le emb :  "+pdbname)
        c1_all=torch.load('./data/lefeature/'+pdbname+'_'+clist[0]+'.pth')
        c2_all=torch.load('./data/lefeature/'+pdbname+'_'+clist[1]+'.pth')
        c1_all=F.max_pool1d(c1_all,16,16)
        c2_all=F.max_pool1d(c2_all,16,16)
        
        node_feature={}
        for v in cl1:
            reduise=v.split('_')[1]
            index=int(reduise[1:])-start1
            node_feature[v]=[]
            node_feature[v].append(c1_all[index].tolist())
            node_feature[v].append(esm_c1_all[index].tolist())
            node_feature[v].append(resFeature[reduise[0]])
            
        for v in cl2:
            reduise=v.split('_')[1]
            index=int(reduise[1:])-start2
            node_feature[v]=[]
            node_feature[v].append(c2_all[index].tolist())
            node_feature[v].append(esm_c2_all[index].tolist())
            node_feature[v].append(resFeature[reduise[0]])
            
        node_features, edge_index,edge_attr=generate_residue_graph(pdbname,node_feature,connect,args.padding)
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index=torch.tensor(edge_index,dtype=torch.long).t().contiguous()
        edge_attr=torch.tensor(edge_attr,dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr,y=complexdict[pdbname])
                
        featureList.append(data) 
        labelList.append(complexdict[pdbname])
        
    
    Affinity_model=AffinityNet(num_features=args.dim,hidden_channel=args.dim,out_channel=args.dim,device=args.device)
    # Affinity_model.load_state_dict(torch.load("./models/saved/gcn/affinity_model2.pt"))
    # Affinity_model=torch.load("./models/saved/gcn/affinity_model2.pt")
    dataset=MyGCNDataset(featureList)
    dataloader=DataLoader(dataset,batch_size=32,shuffle=True)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(Affinity_model.parameters(), lr = 1e-4, weight_decay = 1e-3)

    
    test_prelist, test_truelist,test_loss = gcn_train(Affinity_model,dataloader,optimizer,criterion,args.device,i,0,args.outDir,args.num_layers)
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
        