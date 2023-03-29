import torch
from utils.parse_args import get_args

def to_padding(featureList):
    args=get_args()
    #dis=8  maxlen=71
    #dis=12 maxlen=179
    for i in range(len(featureList)):
        for j in range(2):
            # print(featureList[i][j].shape)
            paddinglen=0
            if featureList[i][j].size(0)<=args.padding:
                paddinglen=args.padding-featureList[i][j].size(0)
                padding=torch.zeros(paddinglen,args.dim)
                padding=padding.to(args.device)
                featureList[i][j]=torch.cat((featureList[i][j],padding),dim=0)
    return featureList