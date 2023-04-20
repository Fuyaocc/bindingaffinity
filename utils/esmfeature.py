import esm
import numpy as np
import torch
import Bio
from Bio.PDB.PDBParser import PDBParser
import os
import logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
import esm.inverse_folding
from parse_args import get_args
import shutil
import gc

def runesm1v(seq, model, alphabet, batch_converter, device):
    res=[]
    data = [("tmp", seq),]
    _, _, batch_tokens = batch_converter(data)
    for i in range(len(seq)):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0][i+1]=alphabet.mask_idx  #mask the residue,32
        # print(batch_tokens_masked)
        with torch.no_grad():
            x=model(batch_tokens_masked.to(device))
        res.append(x[0][i+1].tolist())
    return torch.Tensor(res).to(device)

# dstpath 目的地址
def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        shutil.move(srcfile,dstfile)          #移动文件
        print("move %s -> %s"%( srcfile,dstfile))

def mycopyfile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        shutil.copyfile(srcfile,dstfile)      #复制文件
    

if __name__ == '__main__':
    args=get_args()


    # logging.info("esm1v model loading")
    # esm1v_model_location="/home/ysbgs/xky/esm1v_t33_650M_UR90S_1"
    # esm1v_model, alphabet = esm.pretrained.load_model_and_alphabet(esm1v_model_location)
    # batch_converter = alphabet.get_batch_converter()
    # esm1v_model.to(args.device)
    # logging.info("esm1v model load finish")
    
    # print(args)

    # complexdict={} # pdbname : [seq1, seq2, bingding_affinity]
    # seqdict={}
    # ml=0
    # for line in open(args.inputDir+'benchmark_seq.txt'):
    #     blocks=re.split('\t|\n',line)
    #     seqdict[blocks[0]]=blocks[1]
    #     if len(blocks[1])>ml:
    #         ml=len(blocks[1])
    # print(ml)
# structure
    # for line in open(args.inputDir+'SKEMPI_all_dg_avg.txt'):
    #     blocks=re.split('\t|\n',line)
    #     pdbname=blocks[0]+'+'+blocks[1]
    #     data=[]
    #     data.append(seqdict[blocks[0]])
    #     data.append(seqdict[blocks[1]])
    #     data.append(float(blocks[2]))
    #     complexdict[pdbname]=data

    # for chain_name in seqdict.keys():
    #     logging.info("generate sequcence esm1v : "+chain_name)
        
    #     chain_esm = runesm1v(seqdict[chain_name], esm1v_model, alphabet, batch_converter, args.device)
    #     torch.save(chain_esm.to(torch.device('cpu')),'../data/esmfeature/skempi/'+chain_name+'.pth')
    
    files= os.listdir("../data/esmfeature/strute_emb/")
    
    esm_dict=set()
    for line in files:
        esm_dict.add(line.split("_")[0])
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    for param in model.parameters():
        param.requires_grad = False
    model = model.eval()
    pdbs=[]
    # pdb_dict={}
    with open('../data/pdbbind_data.txt','r') as f1:
        for line in f1:
            pdbs.append(line.split('\t')[0])
            # pdb_dict[line.split('\t')[0]]=line.split('\t')[2]
            
    # f=open('../data/pdbbind_affinity_all2_v2.txt','w')
    
    path1='../data/pdbs/'
    # for name in pdbs:
    #     mycopyfile(sourcepath+name,path+name)
    sum=0
    too_long=["1ogy","1tzn","2wss","3lk4","3sua","3swp"]
    for file in pdbs: #遍历文件夹
        # if(file!='3k8p'):   continue
        if file in esm_dict: continue
        if file in too_long: continue
        fp1 = path1+file+'.pdb'
        parser = PDBParser()
        structure = parser.get_structure("temp", fp1)
        chainGroup=set()
        logging.info(file)
        for chain in structure.get_chains():
            chainGroup.add(chain.get_id())
            
        structure = esm.inverse_folding.util.load_structure(fp1, list(chainGroup))
        coords, _ = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)
        for chain_id in chainGroup:
            rep = esm.inverse_folding.multichain_util.get_encoder_output_for_complex(model, alphabet, coords,chain_id)
            # print(rep.shape)
            torch.save(rep.to(torch.device('cpu')),'../data/esmfeature/strute_emb/'+file.split('.')[0]+'_'+chain_id+'.pth')
            del rep 
            gc.collect()
        
