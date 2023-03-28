import torch
from prodesign.common.protein import from_pdb_string
from prodesign.config import args
from prodesign.model.prodesign import ProDesign
from prodesign.data.dataset import select_residue
from prodesign.common import residue_constants
from utils.getInterfaceRate import getInterfaceRateAndSeq
import numpy as np
from copy import deepcopy
import logging
import os
from Bio.PDB.PDBParser import PDBParser
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
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
    for k in ret0.keys():
        ret0[k]=torch.cat((ret0[k],ret1[k]),dim=0)
    return ret0


def env2prodesign( model,pdb,selects,device):
    info = 'msa'
    israndom = False
    save_T = [1, 2, 5, 10]
    num = 100
    pdb_name =pdb.split("/")[-1]
    fasta_str = ''
    model.eval()
    
    chains=list(selects.keys())
    ret0 = get_feature(pdb, chain_id=chains[0], device=device)
    ret1 = get_feature(pdb, chain_id=chains[0], device=device)
    default_ret=cat_feature(ret0,ret1)
    
    res = {}
    fill = torch.zeros([1, 21], dtype=torch.float)
    fill = fill.to(device)  # 遇到残基为X时，设为0向量
    z = torch.zeros([1, 1], dtype=torch.float)
    z = z.to(device)
    with torch.no_grad():
        i=0
        for chain_id, value in selects.items():            
            all_features = []
            logging.info('prodesign make pred :' + pdb_name  + '_' + chain_id )
            head=0
            for j in range(0, len(value)):  # 序列每个位置都进行预测
                if value[j] != 'X':
                    ret = update_feature(deepcopy(default_ret), i, israndom=israndom)  # 选择残基的局部环境信息
                    assert ret != default_ret
                    preds = model(ret)
                    preds = preds.to(device)
                    if j == 0 and head == 0:
                        all_features.append(preds)
                    else:
                        all_features[0] = torch.cat((all_features[0], preds), dim=0)
                else:
                    if j == 0 and head == 0:
                        all_features.append(fill)
                    else:
                        all_features[0] = torch.cat((all_features[0], fill), dim=0)
                i=i+1
            for idx in range(0, len(value)):
                name = pdb_name  + '_' + chain_id + '-' + str(idx + value[1] - 1) + value[0][idx]
                tmp = deepcopy(all_features)
                tmp[0] = tmp[0][torch.arange(tmp[0].size(0)) != idx]
                res[name] = torch.stack(all_features)


def main(pdb_file):
    num=3
    total_step=2000
    save_step=1000
    pdb_file= args.pdb_file
    fixed=[]
    default_ret = get_feature(pdb_file,device=args.device)
    default_seq= from_aatype_to_strtype(default_ret['seq'])
    print('True seq:\n',default_seq)
    # str_to_fasta(ret['str_seq'],f'4eul')

    model = ProDesign(dim=args.dim,device=args.device)
    model.load_state_dict(torch.load("model89.pt",map_location=args.device))
    
    fasta_str=''
    model.eval()
    with torch.no_grad():
        for i in  range(num):
            ret =  update_feature(deepcopy(default_ret), is_begin = True)
            assert ret!=default_ret
            # str_to_fasta(from_aatype_to_strtype(ret['seq']),f'num{i}_init')
            init_seq=from_aatype_to_strtype(ret['seq'])
            print('Init seq:\n',init_seq)
            # print('identity',get_identity(default_seq,init_seq))

            for step in range(total_step):
                loss,preds,nature_preds=model(ret)
                res=preds[0].argmax(-1).item()
                # res=torch.multinomial(torch.softmax(preds,-1),1).item()
                idx=ret['select_idx']
                #if (ret['seq'][idx]).item()!=res:
                #    print('index %d change from %s to %s'%(idx,(ret['seq'][idx]).item(),res))
                ret['seq'][idx]=res
                update_feature(ret)
                if (step+1)%save_step==0:
                    str_seq=from_aatype_to_strtype(ret['seq'])
                    identity=get_identity(default_seq,str_seq)
                    fasta_str+=f'> {pdb_file} {args.save_file}_num{i}_step{step+1} identity:{identity}\n{str_seq}\n'
                    print(f'num {i} step {step+1} identity {identity}\n',str_seq)

    str_to_fasta(fasta_str,f'{pdb_file} {args.save_file}_{num}')


if __name__=='__main__':
    
    path="/home/ysbgs/xky/cleandata/"
    files= os.listdir(path)
    for file in files:
        if not os.path.isdir(file): 
            fp=path+file
            seqdict,_,_=getInterfaceRateAndSeq(fp,12)
            
            
            
   
    model = ProDesign(dim=args.dim,device="cuda:1")
    model.load_state_dict(torch.load("./model89.pt",map_location="cuda:1"))
    # pdb='/home/ysbg/xky'
    # env2prodesign(model)
