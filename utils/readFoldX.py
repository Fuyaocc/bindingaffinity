import re
import os

def readFoldXResult(path,pdbname):
    foldxres=[0.]*25
    pdb_path=path+"Interaction_"+pdbname+"_AC.fxout"
    if os.path.exists(pdb_path)==False:
        os.system("cd /home/ysbgs/xky/bindingaffinity/data/pdbs/"+"&&"+f'./foldx_20231231 --command=AnalyseComplex --pdb="{pdbname}.pdb" --complexWithDNA=false  --output-dir="/home/ysbgs/xky/foldxResult/"')
    with open(pdb_path,"r") as f:
        for line in f:
            index=0
            if line.startswith("./"):
                energy=re.split("\t|\n",line)
                for i in range(len(energy)):
                    if i<6 or i==27 or i==32 :continue
                    foldxres[index]+=float(energy[i])
                    index+=1
    return foldxres