import torch 
from  torch.utils.data import Dataset

def pickfold(featurelist,labelList,train_index, test_index):
    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]
    for index in train_index:
        x_train.append(featurelist[index])
        y_train.append(labelList[index])
    for index in test_index:
        x_test.append(featurelist[index])
        y_test.append(labelList[index])
    return  x_train,y_train,x_test,y_test

class MyDataset(Dataset):
    def __init__(self,featurelist,labelList):
        self.featurelist=featurelist
        self.labelList=labelList
    
    def __getitem__(self,index):
        item=[]
        item.append(self.featurelist[index][0])
        item.append(self.featurelist[index][1])
        item.append(self.labelList[index])
        return  item

    def __len__(self): 
        return len(self.labelList)
