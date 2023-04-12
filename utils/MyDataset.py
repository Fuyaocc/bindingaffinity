from  torch.utils.data import Dataset
from torch_geometric.data import Data

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
    

def gcn_pickfold(featurelist,train_index, test_index):
    train=[]
    test=[]
    for index in train_index:
        train.append(featurelist[index])
    for index in test_index:
        test.append(featurelist[index])
    return  train,test
    
class MyGCNDataset(Dataset):
    def __init__(self, graph_list):
        self.graph_list = graph_list

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        graph = self.graph_list[idx]
        node_features = graph.x
        edges = graph.edge_index
        edge_attr=graph.edge_attr
        y = graph.y

        # 将数据转换为 PyTorch Geometric 的 Data 对象
        data = Data(x=node_features, edge_index=edges,edge_attr=edge_attr,y=y)

        return data
