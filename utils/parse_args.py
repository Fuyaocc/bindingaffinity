import argparse
import os
import sys
import logging

def get_args():
    """
    :return:进行参数的解析
    """
    description = "you should add those parameter"                   
    parser = argparse.ArgumentParser(description=description)    
    # parser.add_argument('--pdb',required=True,help="The path of structure, e.g. tmp/1CSE.pdb")
    parser.add_argument("--outDir",default="./tmp/",help="Output directory, defaults to tmp")
    parser.add_argument('--device', type=str, default='cuda:1',help='set device')
    parser.add_argument('--inputDir',type=str,default="./data/",help='input data dictionary')
    parser.add_argument('--dim',type=int,default=1280,help='model input dim')
    parser.add_argument('--epoch',type=int,default=500,help='trainning epoch')
    parser.add_argument('--batch_size',type=int,default=32,help='batch size ')
    parser.add_argument('--num_layers',type=int,default=3,help='conv and gru numbers')
    parser.add_argument('--padding',type=int,default=180,help='make feature same length')
    args = parser.parse_args()
    # if not os.path.exists(args.pdb):   #pdb存在
    #     logging.error("structure does not exist under path {}".format(args.pdb))
    #     sys.exit()
    # if not os.path.exists(args.outDir): #out-dir不存在
    #     logging.info("mkdir {} for output".format(args.outDir))
    #     os.mkdir(args.outDir)
    return args

if __name__ == '__main__':
    args = get_args()
    print(args)