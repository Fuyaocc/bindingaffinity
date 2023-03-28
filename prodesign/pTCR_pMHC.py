import pandas
import torch
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
from prodesign.common.protein import from_pdb_string
from prodesign.common import residue_constants
# from utils.config import args
from prodesign.model.prodesign import ProDesign
import numpy as np
from copy import deepcopy
import re
from einops import rearrange
import torch.nn.functional as F
from prodesign.model.functional import rigids_from_3x3
import time

from sklearn import metrics


# import matplotlib.pyplot as plt

def select_residue(ret, pos, cut_off=20):
    # logging.info("start select_residue")
    # select residue
    assert ret['seq'].size(0) == len(ret['str_seq'])
    nei_mask = None
    ca_idx = residue_constants.atom_order['CA']
    # noX_idx=torch.arange(ret['seq'].size(0))[ret['seq']!=residue_constants.unk_restype_index]
    maxstep = 100
    i = 0
    # logging.info(str(pos))
    while i < maxstep and (nei_mask is None or nei_mask.sum() < 2):
        i = i + 1
        select_idx = pos
        nei_mask = torch.sqrt(
            ((ret['coord'][..., ca_idx, :] - ret['coord'][..., select_idx, ca_idx, :]) ** 2).sum(-1)) <= cut_off
        if 'coord_mask' in ret:
            ret['mask'] = torch.sum(ret['coord_mask'], dim=-1) > 0
            nei_mask = nei_mask * ret['mask']

    nei_mask[select_idx] = False
    nei_type = F.one_hot(ret['seq'], 21)  # X

    nei_idx = torch.arange(ret['seq'].size(0), device=ret['seq'].device) - select_idx
    target_R = ret["R"][..., select_idx, :, :]
    target_t = ret["t"][..., select_idx, :]
    nei_R = torch.einsum('... d h, ... d w -> ... w h', target_R, ret["R"])
    nei_t = torch.einsum('... d w, ... d -> ... w', ret["R"], target_t - ret["t"])
    nei_feature = torch.cat((nei_type, rearrange(nei_R, '... h w -> ... (h w)'), nei_t, nei_idx[:, None]), dim=-1)
    ret['nei_feature'] = nei_feature.masked_select(nei_mask[:, None]).reshape((-1, nei_feature.size(-1)))
    ret['nei_mask'] = torch.ones(nei_mask.sum(), device=nei_mask.device)
    ret['select_idx'] = select_idx
    ret['label'] = ret['seq'][select_idx]
    return ret


def get_identity(a, b):
    assert len(a) == len(b)
    identity = sum(a[i] == b[i] for i in range(len(a))) / len(a)
    return identity


def get_feature(pdb_file, chain_id=None, device='cpu'):
    '''
     for design
    '''
    with open(pdb_file, 'r') as f:
        pdb_str = f.read()
    protein = from_pdb_string(pdb_str, chain_id)
    seq = torch.tensor(protein.aatype, dtype=torch.int64).to(device)
    coord = torch.from_numpy(protein.atom_positions).to(device)
    coord_mask = torch.from_numpy(protein.atom_mask).to(device)
    n_idx = residue_constants.atom_order['N']
    ca_idx = residue_constants.atom_order['CA']
    c_idx = residue_constants.atom_order['C']
    assert coord.shape[-2] > min(n_idx, ca_idx, c_idx)
    R, t = rigids_from_3x3(coord, indices=(c_idx, ca_idx, n_idx))
    ret = dict(seq=seq, str_seq=protein.str_aatype, coord=coord, coord_mask=coord_mask, R=R, t=t)
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


def str_to_fasta(fasta_str, des):
    with open(f'{args.fasta_dir}/{des}.fasta', 'w') as f:
        f.write(fasta_str)


def featurematch(pred):
    # 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'
    le_c = [10, 0, 7, 19, 15, 6, 1, 16, 9, 3, 14, 11, 5, 2, 13, 18, 12, 8, 17, 4]
    # ems_c={'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F','Y', 'M', 'H', 'W', 'C'}
    x = pred.tolist()
    feature = []
    y = []
    for i in range(len(x[0])):
        y.append(x[0][le_c[i]])
    y = torch.tensor(y)
    feature.append(y)
    feature = torch.stack(feature)
    return feature


def env2prodesign( model,pdb,selects,device):
    info = 'msa'
    israndom = False
    save_T = [1, 2, 5, 10]
    num = 100
    pdb_name =pdb.split("/")[-1]
    fasta_str = ''
    model.eval()

    res = {}
    fill = torch.zeros([1, 21], dtype=torch.float)
    fill = fill.to(device)  # 遇到残基为X时，设为0向量
    z = torch.zeros([1, 1], dtype=torch.float)
    z = z.to(device)
    with torch.no_grad():
        for chain_id, value in selects.items():
            default_ret = get_feature(pdb, chain_id=chain_id, device=device)
            

            all_features = []
            i = -1
            logging.info('prodesign make pred :' + pdb_name  + '_' + chain_id )
            head = 0
            for j in range(0, value[1] - 1):
                if j == 0:
                    all_features.append(fill)
                else:
                    all_features[0] = torch.cat((all_features[0], fill), dim=0)
                head = 1

            for j in range(0, len(value[0])):  # 序列每个位置都进行预测
                if value[0][j] != 'X':
                    i = i + 1
                    ret = update_feature(deepcopy(default_ret), i, israndom=israndom)  # 选择残基的局部环境信息
                    assert ret != default_ret
                    preds = model(ret)
                    preds = featurematch(preds)  # 保证氨基酸顺序和esm1v的结果顺序一致(便于替换) ， 可以不使用
                    preds = preds.to(device)
                    preds = torch.cat((preds, z), dim=1)  # 补全21维， le的模型只有输出20维，最后一维为X
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
            for idx in range(0, len(value[0])):
                name = pdb_name + '_' + more_name + '_' + chain_id + '-' + str(idx + value[1] - 1) + value[0][idx]
                tmp = deepcopy(all_features)
                tmp[0] = tmp[0][torch.arange(tmp[0].size(0)) != idx]
                res[name] = torch.stack(all_features)
    return res


import os
from Bio import PDB
import pandas as pd
import random
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model


def random_data(tcr, l, num):
    t_re = []
    for t in tcr:
        if len(t) == l:
            t_re.append(t)
    random.shuffle(t_re)
    return t_re[0:num]


def pdb_one_chain(dir, out_file):
    parser = PDB.PDBParser()
    structure = parser.get_structure('2FH7', dir)
    mm = structure[0]
    max_name = ""
    min_name = ""
    max_num = 0
    min_num = 10000
    ppb = PDB.PPBuilder()
    chainlist = []
    for chain in mm:
        print(len(chain))
        chainlist.append(chain.id)
        if len(chain) < 5:
            continue
        if len(chain) > max_num:
            max_name = chain.id
            max_num = len(chain)
        if len(chain) < min_num:
            min_name = chain.id
            min_num = len(chain)

    print(min_name)
    chain_Pep = mm[min_name]
    model_nr = 1
    re_Pep = ppb.build_peptides(chain_Pep, model_nr)[0].get_sequence()
    if len(re_Pep) > 12:
        return re_Pep, min_name
    print(re_Pep)
    chainlist.remove(min_name)

    id = chain_Pep.child_list[len(chain_Pep) - 1].id[1]
    for chain_id in chainlist:
        chain_P = mm[chain_id]
        for amino in chain_P.child_list:
            id += 1
            amino.parent = None
            amino.id = (' ', id, ' ')
            chain_Pep.add(amino)

    structure = Structure("test.py")
    mo = Model(0)
    mo.add(chain_Pep)
    structure.add(mo)
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(out_file)
    return re_Pep, min_name
