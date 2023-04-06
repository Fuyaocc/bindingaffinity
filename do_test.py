import pandas as pd
import re

def check_len(i,x_train,x_test):
    with open(f"./tmp/local_val{i}.txt",'w') as f:
        f.write('train\n')
        train_x1_av,train_x2_av=0.,0.
        for x in x_train:
            f.write(x[2])
            f.write('\t')
            f.write(str(len(x[0])))
            f.write('\t')
            f.write(str(len(x[1])))
            f.write('\n')
            train_x1_av=train_x1_av+len(x[0])
            train_x2_av=train_x2_av+len(x[1])
        train_x1_av=train_x1_av/len(x_train)
        train_x2_av=train_x2_av/len(x_train)
        
        
        test_x1_av,test_x2_av=0.,0.
        f.write('\ntest\n')
        for x in x_test:
            f.write(x[2])
            f.write('\t')
            f.write(str(len(x[0])))
            f.write('\t')
            f.write(str(len(x[1])))
            f.write('\n')
            test_x1_av=test_x1_av+len(x[0])
            test_x2_av=test_x2_av+len(x[1])
        test_x1_av=test_x1_av/len(x_test)
        test_x2_av=test_x2_av/len(x_test)
        f.write("train\tchain_0  len_avg = "+str(train_x1_av)+" , "+"chain_1  len_avg = "+str(train_x2_av)+ " , "+"sum len_avg = "+str(train_x1_av+train_x2_av))
        f.write("\ntest\tchain_0  len_avg = "+str(test_x1_av)+" , "+"chain_1  len_avg = "+str(test_x2_av)+ " , "+"sum len_avg = "+str(test_x1_av+test_x2_av))
    
def calpcc():
    t=[]
    p=[]
    with open("./tmp/test/val1/test_400.txt","r") as f:
        for line in f:
            x=re.split("\t\t|\n",line)
            t.append(float(x[0]))
            p.append(float(x[1]))
    df = pd.DataFrame({'label':t, 'pre':p})
    test_pcc = df.pre.corr(df.label)
    print(test_pcc)

if __name__ == "__main__":
    calpcc()
        
