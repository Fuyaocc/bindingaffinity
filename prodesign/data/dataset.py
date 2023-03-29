import contextlib
import functools
import logging
import math
from io import BytesIO
import pathlib
import random
import zipfile

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import WeightedRandomSampler
from einops import rearrange

from prodesign.common import protein, residue_constants
from prodesign.data.parsers import parse_a3m, parse_fasta
from prodesign.model.features import FeatureBuilder
from prodesign.utils import default, exists
from prodesign.model.functional import rigids_from_3x3
# from utils.design import get_feature

logger = logging.getLogger(__file__)

class ProteinSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, descriptions = None):
        self.sequences = sequences
        self.descriptions = descriptions
        assert not exists(self.descriptions) or len(self.sequences) == len(self.descriptions)

    def __getitem__(self, idx):
        input_sequence = self.sequences[idx]
        seq = torch.tensor(residue_constants.sequence_to_onehot(
            sequence=input_sequence,
            mapping=residue_constants.restype_order_with_x,
            map_unknown_to_x=True), dtype=torch.int).argmax(-1)
        #residue_index = torch.arange(len(input_sequence), dtype=torch.int)
        str_seq = ''.join(map(lambda a: a if a in residue_constants.restype_order_with_x else residue_constants.restypes_with_x[-1], input_sequence))
        mask = torch.ones(len(input_sequence), dtype=torch.bool)
        return dict(pid=self.descriptions[idx] if self.descriptions else str(idx),
                seq=seq,
                str_seq=str_seq,
                mask=mask)

    def __len__(self):
        return len(self.sequences)

    def collate_fn(self, batch, feat_builder=None):
        fields = ('pid', 'seq', 'mask', 'str_seq')
        pids, seqs, masks, str_seqs = list(zip(*[[b[k] for k in fields] for b in batch]))
        lengths = tuple(len(s) for s in str_seqs)
        max_batch_len = max(lengths)

        padded_seqs = pad_for_batch(seqs, max_batch_len, 'seq')
        padded_masks = pad_for_batch(masks, max_batch_len, 'msk')

        ret = dict(pid=pids,
                seq=padded_seqs,
                mask=padded_masks,
                str_seq=str_seqs)

        if feat_builder:
            ret = feat_builder.build(ret)

        return ret

class ProteinStructureDataset(torch.utils.data.Dataset):
    FEAT_PDB = 0x01
    FEAT_MSA = 0x02
    FEAT_ALL = 0xff

    global features

    def __init__(self, data_dir,device='cpu', data_idx='name.idx', max_seq_len=None, coord_type='npz', feat_flags=FEAT_ALL&(~FEAT_MSA)):
        super().__init__()

        self.data_dir = pathlib.Path(data_dir)
        if zipfile.is_zipfile(self.data_dir):
            self.data_dir = zipfile.ZipFile(self.data_dir)
        self.max_seq_len = max_seq_len
        self.feat_flags = feat_flags
        logger.info('load idx data from: %s', data_idx)
        with self._fileobj(data_idx) as f:
            self.pids = list(map(lambda x: self._ftext(x).strip().split(), f))

        self.mapping = {}
        if self._fstat('mapping.idx'):
            with self._fileobj('mapping.idx') as f:
                for line in filter(lambda x: len(x)>0, map(lambda x: self._ftext(x).strip(), f)):
                    v, k = line.split()
                    self.mapping[k] = v

        self.resolu = {}
        if self._fstat('resolu.idx'):
            with self._fileobj('resolu.idx') as f:
                for line in filter(lambda x: len(x)>0, map(lambda x: self._ftext(x).strip(), f)):
                    k, v = line.split()
                    self.resolu[k] = float(v)

        self.FASTA = 'fasta'

        self.PDB = coord_type
        for t in ('npz', 'coord'):
            if self.PDB is None and self._fstat(f'{t}/'):
                self.PDB = t
                break
        logger.info('load structure data from: %s', self.PDB)
        assert not (feat_flags & ProteinStructureDataset.FEAT_PDB) or (self.PDB is not None)
        
        self.device=device

    def __getstate__(self):
        logger.debug('being pickled ...')
        d = self.__dict__
        if isinstance(self.data_dir, zipfile.ZipFile):
            d['data_dir'] = self.data_dir.filename
        return d

    def __setstate__(self, d):
        logger.debug('being unpickled ...')
        if zipfile.is_zipfile(d['data_dir']):
            d['data_dir'] = zipfile.ZipFile(d['data_dir'])
        self.__dict__ = d


    def __getitem__(self, idx):
        pids = self.pids[idx]
        pid = pids[np.random.randint(len(pids))]

        pkey = self.mapping[pid] if pid in self.mapping else pid
        seq_feats = self.get_seq_features(pkey)
        
        ret = dict(pid=pid, resolu=self.get_resolution(pid), **seq_feats)
        if self.feat_flags & ProteinStructureDataset.FEAT_PDB:
            assert self.PDB in ('npz',)
            if self.PDB == 'npz':
                ret.update(self.get_structure_label_npz(pid, seq_feats['str_seq']))
        
        ret=select_residue(ret)
        return ret

    def __len__(self):
        return len(self.pids)

    @contextlib.contextmanager
    def _fileobj(self, filename:str):
        if filename.startswith('/'):
            with open(filename, mode='rb') as f:
                yield f
        elif isinstance(self.data_dir, zipfile.ZipFile):
            with self.data_dir.open(filename, mode='r') as f:
                yield f
        else:
            with open(self.data_dir / filename, mode='rb') as f:
                yield f
    
    def __len__(self):
        return len(self.pids)

    def _fstat(self, filename):
        if isinstance(self.data_dir, zipfile.ZipFile):
            try:
                self.data_dir.getinfo(filename)
                return True
            except KeyError as e:
                return False
        return (self.data_dir / filename).exists()

    def _ftext(self, line, encoding='utf-8'):
        if isinstance(line, bytes):
            return line.decode(encoding)
        return line

    def get_resolution(self, protein_id):
        pid = protein_id.split('_')
        return self.resolu.get(pid[0], None)

    def get_structure_label_npz(self, protein_id, str_seq):
        if self._fstat(f'{self.PDB}/{protein_id}.npz'):
            with self._fileobj(f'{self.PDB}/{protein_id}.npz') as f:
                structure = np.load(BytesIO(f.read()),allow_pickle=True)
                ret = dict(
                        coord=torch.from_numpy(structure['coord']).to(self.device),
                        coord_mask=torch.from_numpy(structure['coord_mask']).to(self.device))
                if 'bfactor' in structure:
                    ret.update(coord_plddt=torch.from_numpy(structure['bfactor']).to(self.device))
                return ret
        return dict()

    def get_seq_features(self, protein_id):
        """Runs alignment tools on the input sequence and creates features."""
        with self._fileobj(f'{self.FASTA}/{protein_id}.fasta') as f:
            input_fasta_str = self._ftext(f.read())
        input_seqs, input_descs = parse_fasta(input_fasta_str)
        if len(input_seqs) != 1:
            raise ValueError(
                f'More than one input sequence found in {self.FASTA}/{protein_id}.fasta .')
        input_sequence = input_seqs[0]
        input_description = input_descs[0]

        input_sequence = input_sequence[:self.max_seq_len]

        seq = torch.tensor(residue_constants.sequence_to_onehot(
            sequence=input_sequence,
            mapping=residue_constants.restype_order_with_x,
            map_unknown_to_x=True), dtype=torch.int,device=self.device).argmax(-1)
        #residue_index = torch.arange(len(input_sequence), dtype=torch.int)
        str_seq = ''.join(map(lambda a: a if a in residue_constants.restype_order_with_x else residue_constants.restypes_with_x[-1], input_sequence))
        mask = torch.ones(len(input_sequence), dtype=torch.bool,device=self.device)

        return dict(seq=seq,
                str_seq=str_seq,
                mask=mask)

    def collate_fn(self, batch, feat_builder=None):
        fields = ('pid', 'resolu', 'seq', 'mask', 'str_seq','label','nei_feature','nei_mask')
        pids, resolutions, seqs, masks, str_seqs,label,nei_feature,nei_mask = list(zip(*[[b[k] for k in fields] for b in batch]))
        lengths = tuple(len(s) for s in str_seqs)
        max_batch_len = max(lengths)

        padded_seqs = pad_for_batch(seqs, max_batch_len, 'seq')
        padded_masks = pad_for_batch(masks, max_batch_len, 'msk')
        
        nei_max_batch_len=max(tuple(f.size(0) for f in nei_feature))
        padded_nei_feature=pad_for_batch(nei_feature,nei_max_batch_len, 'nei')
        padded_nei_mask=pad_for_batch(nei_mask,nei_max_batch_len, 'nei_msk')
        
        label=torch.tensor(label,device=self.device) 
        ret = dict(pid=pids,
                resolution=resolutions,
                seq=padded_seqs,
                mask=padded_masks,
                label=label,
                nei_feature=padded_nei_feature,
                nei_mask=padded_nei_mask,
                str_seq=str_seqs)

        #if self.feat_flags & ProteinStructureDataset.FEAT_PDB:
        #    fields = ('coord', 'coord_mask')
        #    coords, coord_masks = list(zip(*[[b[k] for k in fields] for b in batch]))

        #    padded_coords = pad_for_batch(coords, max_batch_len, 'crd')
        #    padded_coord_masks = pad_for_batch(coord_masks, max_batch_len, 'crd_msk')
        #    ret.update(
        #        coord=padded_coords,
        #        coord_mask=padded_coord_masks)

        if feat_builder:
            ret = feat_builder.build(ret)

        return ret

def pad_for_batch(items, batch_length, dtype):
    """Pad a list of items to batch_length using values dependent on the item type.

    Args:
        items: List of items to pad (i.e. sequences or masks represented as arrays of
            numbers, angles, coordinates, pssms).
        batch_length: The integer maximum length of any of the items in the input. All
            items are padded so that their length matches this number.
        dtype: A string ('seq', 'msk', 'crd') reperesenting the type of
            data included in items.

    Returns:
         A padded list of the input items, all independently converted to Torch tensors.
    """
    batch = []
    if dtype == 'seq':
        for seq in items:
            z = torch.ones(batch_length - seq.shape[0], dtype=seq.dtype,device=seq.device) * residue_constants.unk_restype_index
            c = torch.cat((seq, z), dim=0)
            batch.append(c)
    elif dtype == 'msk':
        # Mask sequences (1 if present, 0 if absent) are padded with 0s
        for msk in items:
            z = torch.zeros(batch_length - msk.shape[0], dtype=msk.dtype,device=msk.device)
            c = torch.cat((msk, z), dim=0)
            batch.append(c)
    elif dtype == 'crd':
        for item in items:
            z = torch.zeros((batch_length - item.shape[0],  item.shape[-2], item.shape[-1]), dtype=item.dtype, device=item.device)
            c = torch.cat((item, z), dim=0)
            batch.append(c)
    elif dtype == 'crd_msk' or dtype == 'coord_plddt':
        for item in items:
            z = torch.zeros((batch_length - item.shape[0],  item.shape[-1]), dtype=item.dtype, device=item.device)
            c = torch.cat((item, z), dim=0)
            batch.append(c)
    elif dtype == 'nei':
        for item in items:
            z = torch.zeros((batch_length - item.shape[0], item.shape[-1]), dtype=item.dtype, device=item.device)
            c = torch.cat((item, z), dim=0)
            batch.append(c)
    elif dtype == 'nei_msk':
        for item in items:
            z = torch.zeros(batch_length - item.shape[0], dtype=item.dtype, device=item.device)
            c = torch.cat((item, z), dim=0)
            batch.append(c)
    else:
        raise ValueError('Not implement yet!')
    batch = torch.stack(batch, dim=0)
    return batch
    

def select_residue(ret,cut_off=15):
        # select residue
        fields=('seq','coord','coord_mask')

        assert ret['seq'].size(0)==len(ret['str_seq'])
        nei_mask=None
        n_idx = residue_constants.atom_order['N']
        ca_idx = residue_constants.atom_order['CA']
        c_idx = residue_constants.atom_order['C']
        assert ret['coord'].shape[-2] > min(n_idx, ca_idx, c_idx)
        noX_idx=torch.arange(ret['seq'].size(0))[ret['seq']!=residue_constants.unk_restype_index]
        while nei_mask is None or nei_mask.sum()<1:
            select_idx=np.random.choice(noX_idx)
            nei_mask=torch.sqrt(((ret['coord'][...,ca_idx,:]-ret['coord'][...,select_idx,ca_idx,:])**2).sum(-1))<=cut_off
            if 'coord_mask' in ret:
                ret['mask'] = torch.sum(ret['coord_mask'], dim=-1) > 0
                nei_mask=nei_mask*ret['mask']

        nei_mask[select_idx]=False
        nei_type=F.one_hot(ret['seq'],21) # X

        nei_idx=torch.arange(ret['seq'].size(0),device=ret['seq'].device)-select_idx
        
        R,t = rigids_from_3x3(ret['coord'], indices=(c_idx, ca_idx, n_idx))
        target_R=R[...,select_idx,:,:]
        target_t=t[...,select_idx,:]
        #nei_R=torch.einsum('... d h, ... d w -> ... h w',target_R,R)
        #nei_t=torch.einsum('... d h, ... d -> ... h',target_R,t-target_t)
        nei_R=torch.einsum('... d h, ... d w -> ... w h',target_R,R)
        nei_t=torch.einsum('... d w, ... d -> ... w',R,target_t-t)
        nei_R=nei_R.to(torch.float32)
        nei_t=nei_t.to(torch.float32)
        nei_feature=torch.cat((nei_type,rearrange(nei_R,'... h w -> ... (h w)'),nei_t,nei_idx[:,None]),dim=-1)
        ret['nei_feature']=nei_feature.masked_select(nei_mask[:,None]).reshape((-1,nei_feature.size(-1)))
        ret['nei_mask']=torch.ones(nei_mask.sum(),device=nei_mask.device)
        ret['select_idx']=select_idx
        ret['label']=ret['seq'][select_idx]
        return ret

def load(data_dir,device='cpu',data_idx='name.idx', min_crop_len=None, max_crop_len=None, crop_probability=0, crop_algorithm='random', feats=None, is_training=True, feat_flags=ProteinStructureDataset.FEAT_ALL, **kwargs):
    dataset = ProteinStructureDataset(data_dir, device=device, data_idx=data_idx, feat_flags=feat_flags)
    if not 'collate_fn' in kwargs:
        kwargs['collate_fn'] = functools.partial(dataset.collate_fn,
                feat_builder=FeatureBuilder(feats, is_training=is_training))
    if 'weights' in kwargs:
        weights = kwargs.pop('weights')
        if weights:
            kwargs['sampler'] = WeightedRandomSampler(weights, num_samples=len(weights))
            if 'shuffle' in kwargs:
                kwargs.pop('shuffle')
    return torch.utils.data.DataLoader(dataset, **kwargs)
