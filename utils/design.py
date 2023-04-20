import torch
import numpy as np
from copy import deepcopy
import logging
import os
import re
from Bio.PDB.PDBParser import PDBParser
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
from getInterfaceRate import getInterfaceRateAndSeq
import sys
sys.path.append("../")
from prodesign.common.protein import from_pdb_string
from prodesign.model.prodesign import ProDesign
from prodesign.data.dataset import select_residue
from prodesign.common import residue_constants

def get_identity(a,b):
    assert len(a)==len(b)
    identity=sum(a[i]==b[i] for i in range(len(a)))/len(a)
    return identity
        
def get_feature(pdb_file,chain_id=None,device='cpu'):
    '''
     for design
    '''
    with open(pdb_file,'r') as f:
        pdb_str=f.read()
    protein=from_pdb_string(pdb_str,chain_id)
    seq = torch.tensor(protein.aatype,dtype=torch.int64).to(device)
    coord = torch.from_numpy(protein.atom_positions).to(device)
    coord_mask = torch.from_numpy(protein.atom_mask).to(device)
    ret = dict(seq = seq, str_seq = protein.str_aatype, coord = coord, coord_mask = coord_mask)
    return ret

def update_feature(ret, pos, israndom=False):
    # logging.info("start update_feature")
    fixed = []
    ret = select_residue(ret, pos)
    ret['nei_feature'] = ret['nei_feature'].unsqueeze(0).float()
    ret['nei_mask'] = ret['nei_mask'].unsqueeze(0)
    return ret

def from_aatype_to_strtype(seq):
    restype_idx = ''
    for idx in seq:
        restype_idx = restype_idx + (residue_constants.restypes_with_x[idx])
    return restype_idx

def str_to_fasta(fasta_str,des):
    with open(f'fasta/{des}.fasta','w') as f:
        f.write(fasta_str)

def cat_feature(ret0,ret1):
    print(list(ret0.keys()))
    ret0['seq']=torch.cat((ret0['seq'],ret1['seq']),dim=0)
    ret0['str_seq']=ret0['str_seq']+ret1['str_seq']
    ret0['coord']=torch.cat((ret0['coord'],ret1['coord']),dim=0)
    ret0['coord_mask']=torch.cat((ret0['coord_mask'],ret1['coord_mask']),dim=0)

    return ret0


def env2prodesign( model,pdb,selects,device):
    israndom = False
    pdb_name =pdb.split('/')[-1].split('.')[0]
    model.eval()
    logging.info(pdb_name)
    chains=list(selects.keys())
    ret0 = get_feature(pdb, chain_id=chains[0], device=device)
    selects[chains[0]]=ret0["seq"]
    ret1 = get_feature(pdb, chain_id=chains[1], device=device)
    selects[chains[1]]=ret1["seq"]
    default_ret=cat_feature(ret0,ret1)

    with torch.no_grad():
        i=0
        for chain_id, value in selects.items():          
            logging.info('prodesign make pred :' + pdb_name  + '_' + chain_id )
            head=0
            print(len(value))
            for j in range(0, len(value)):  # 序列每个位置都进行预测
                ret = update_feature(deepcopy(default_ret), i, israndom=israndom)  # 选择残基的局部环境信息
                assert ret != default_ret
                preds = model(ret)
                preds = preds.to(device)
                if j == 0 and head == 0:
                    all_features = preds
                else:
                    all_features = torch.cat((all_features, preds), dim=0)
                head=1
                i=i+1
            print(all_features.shape)
            torch.save(all_features.to(torch.device('cpu')),'../data/lefeature/'+pdb_name+'_'+chain_id+'.pth')


if __name__=='__main__':
    model = ProDesign(dim=256,device="cuda:1")
    model.load_state_dict(torch.load("../prodesign/model89.pt",map_location="cuda:1"))
    path='../data/pdbs/'
    with open("../data/prodesign.txt","r") as f:
        for line in f:
            x=re.split('\t|\n',line)
            selects={}
            selects[x[1].split('_')[0]]=x[2]
            selects[x[1].split('_')[1]]=x[3]
            env2prodesign(model,path+x[0]+'.pdb',selects,device="cuda:1")
