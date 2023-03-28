import functools
from inspect import isfunction

import numpy as np
import torch
from torch.nn import functional as F
from einops import rearrange,repeat

from prodesign.common import residue_constants
from prodesign.model.functional import (
        angles_from_positions,
        batched_gather,
        rigids_from_3x3,
        rigids_from_positions,
        symmetric_ground_truth_create_alt)
from prodesign.utils import default, exists

_feats_fn = {}


def take1st(fn):
    """Supply all arguments but the first."""
    @functools.wraps(fn)
    def fc(*args, **kwargs):
        return lambda x: fn(x, *args, **kwargs)

    global _feats_fn
    _feats_fn[fn.__name__] = fc

    return fc

@take1st
def make_seq_mask(protein, padd_id=20, is_training=True):
    mask = protein['seq'] != padd_id
    protein['mask'] = mask.bool()
    return protein

@take1st
def make_coord_mask(protein, includes=None, excludes=None, is_training=True):
    if exists(includes) or exists(excludes):
        restype_atom14_mask = np.copy(residue_constants.restype_atom14_mask)
        if exists(includes):
            includes = set(includes)
            for i in range(residue_constants.restype_num):
                resname = residue_constants.restype_1to3[residue_constants.restypes[i]]
                atom_list = residue_constants.restype_name_to_atom14_names[resname]
                for j in range(restype_atom14_mask.shape[1]):
                    if restype_atom14_mask[i,j] > 0 and atom_list[j] not in includes:
                        restype_atom14_mask[i,j] = 0
        if exists(excludes):
            excludes = set(excludes)
            for i in range(residue_constants.restype_num):
                resname = residue_constants.restype_1to3[residue_constants.restypes[i]]
                atom_list = residue_constants.restype_name_to_atom14_names[resname]
                for j in range(restype_atom14_mask.shape[1]):
                    if restype_atom14_mask[i,j] > 0 and atom_list[j] in excludes:
                        restype_atom14_mask[i,j] = 0
        coord_exists = batched_gather(restype_atom14_mask,
                protein['seq'])
    else:
        coord_exists = batched_gather(residue_constants.restype_atom14_mask,
                protein['seq'])
    protein['coord_exists'] = coord_exists
    return protein

@take1st
def make_LE(protein,cut_off=12,is_training=True):
    #seq: b l
    #coord: b l 14 3
    idx=protein['select_idx']
    # b l
    nei_mask=torch.sqrt(((protein['coord'][:,:,1,:]-protein['coord'][torch.arange(idx.size(0)),idx.tolist(),1,None])**2).sum(dim=-1))<=cut_off
    #print(nei_mask[1],protein['mask'][1])
    nei_mask=nei_mask*protein['mask']
    
    # b l 1
    nei_idx= torch.arange(protein['seq'].size(-1))
    nei_idx=nei_idx[:,None]-nei_idx[None,:]
    nei_idx=torch.index_select(nei_idx,0,idx).unsqueeze(-1)
    # b l 20
    nei_type=F.one_hot(protein['seq'],21) # 21

    n_idx = residue_constants.atom_order['N']
    ca_idx = residue_constants.atom_order['CA']
    c_idx = residue_constants.atom_order['C']
    assert protein['coord'].shape[-2] > min(n_idx, ca_idx, c_idx)
    # b l l 12
    R,t = rigids_from_3x3(protein['coord'], indices=(c_idx, ca_idx, n_idx))
    #target_R_inverse=torch.inverse(R[torch.arange(idx.size(0)),idx.tolist(),None])
    target_R_T=R[torch.arange(idx.size(0)),idx.tolist(),None].transpose(-1,-2)
    target_t=t[torch.arange(idx.size(0)),idx.tolist(),None]
    # R=R_{j}R_{i}^-1
    # t=t_{j}-Rt_{i}
    nei_R=torch.einsum('... d h, ... w d -> ... h w',target_R_T,R) 
    nei_t=-torch.einsum('... h w, ... h -> ... w',nei_R,target_t)+t
    nei_pos=torch.cat((rearrange(nei_R,'b l h w -> b l (h w)'),nei_t),dim=-1)
    # b l 33
    protein['nei_feature']=torch.cat((nei_type,nei_pos,nei_idx),dim=-1)
    protein['nei_mask']=nei_mask
    protein['labels'] = protein['seq'][torch.arange(idx.size(0)),idx.tolist()]
    
    #CA=protein['coord'][...,1,:]
    # b l l
    #nei_mask=torch.sqrt((CA[:,None,:,:]-CA[:,:,None,:])**2).sum(dim=-1))<=cut_off
    #nei_mask=nei_mask*protein['mask']
    
    # b l l 1
    #nei_idx= torch.arange(protein['seq'].size(-1))
    #nei_idx=repeat(nei_idx[:,None]-nei_idx[None,:],'l l -> b l l ()',b=protein['seq'].size(0))
    # b l l 20
    #nei_type=repeat(F.onehot(protein['seq'],20),'b l o ->b l l o')

    #n_idx = residue_constants.atom_order['N']
    #ca_idx = residue_constants.atom_order['CA']
    #c_idx = residue_constants.atom_order['C']
    #assert protein['coord'].shape[-2] > min(n_idx, ca_idx, c_idx)
    # b l l 12
    #R,t = rigids_from_3x3(protein['coord'], indices=(c_idx, ca_idx, n_idx))
    #R=rearrange(R,'b l c d -> b l (c d)')
    #nei_pos=torch.cat((R,t),dim=-1)
    #nei_pos=nei_pos[...,:,None]-nei_pos[...,None,:]
    # b l l 33
    #protein['nei_feature']=torch.cat((nei_type,nei_pos,nei_idx),dim=-1)
    #protein['nei_mask']=nei_mask
    #random_select_idx(protein)

    return protein

@take1st
def make_seq_idx(protein):
    assert 'seq' in protein
    seq_idx=torch.arange(protein['seq'].size(0))
    protein['seq_idx']=seq_idx[:,None]-seq_idx[None,:]
    return protein

@take1st
def make_backbone_affine(protein):
    n_idx = residue_constants.atom_order['N']
    ca_idx = residue_constants.atom_order['CA']
    c_idx = residue_constants.atom_order['C']

    assert protein['coord'].shape[-2] > min(n_idx, ca_idx, c_idx)
    protein['backbone_affine'] = rigids_from_3x3(protein['coord'], indices=(c_idx, ca_idx, n_idx))
    return protein

@take1st
def make_affine(protein, is_training=True):
    if is_training:
        feats = rigids_from_positions(protein['seq'], protein.get('coord'), protein.get('coord_mask'))
        protein.update(feats)
    return protein

@take1st
def make_torsion_angles(protein, is_training=True):
    if is_training and ('coord' in protein and 'coord_mask' in protein):
        feats = angles_from_positions(protein['seq'], protein['coord'], protein['coord_mask'])
        protein.update(feats)
        #protein['torsion_chi_angles'] = feats['torsion_angles'][...,3:,:]  # (B, N, 7, 2) -> (B, N, 4, 2)
        #protein['torsion_chi_mask'] = feats['torsion_angles_mask'][...,3:]  # (B, N, 7) -> (B, N, 4)
    return protein

@take1st
def make_to_device(protein, fields, device, is_training=True):
    if isfunction(device):
        device = device()
    for k in fields:
        if k in protein:
            protein[k] = protein[k].to(device)
    return protein

@take1st
def make_delete_fields(protein, fields, is_training=True):
    for k in fields:
        if k in protein:
            del protein[k]
    return protein

@take1st
def make_selection(protein, fields, is_training=True):
    return {k: protein[k] for k in fields}

class FeatureBuilder:
    def __init__(self, config, is_training=True):
        self.config = config
        self.training = is_training

    def build(self, protein):
        for fn, kwargs in default(self.config, []):
            f = _feats_fn[fn](is_training=self.training, **kwargs)
            protein = f(protein)
        return protein

    def __call__(self, protein):
        return self.build(protein)
