#读取PDB文件，并拆分为相互作用的两部分
#需要pymol
from pymol import cmd
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import DSSP
import os
import sys
import logging

def splitPDB2Part(pdbPath,savePath,interactionInfo):
    pdbName=os.path.basename(os.path.splitext(pdbPath)[0])
    subunit1,subunit2=interactionInfo.split("_") #subunit1:E链 subunit2:I链
    cmd.delete('all')
    cmd.load(pdbPath)
    cmd.remove('hetatm')  #删除hetatm（删除杂原子）
    names=cmd.get_names()
    if len(names)!=1:
        logging.error("{} have multiple pdb names:{}".format(pdbPath,names))
        sys.exit()
    try:
        #subunit1
        cmd.create('tmp',"chain {}".format(",".join(subunit1)))
        cmd.save(os.path.join(savePath,"{}_{}.pdb".format(pdbName,subunit1)),'tmp')
        cmd.delete("tmp")
    except:
        logging.error("chain {} error in interaction Info {} for pdb {}".format(subunit1,interactionInfo,pdbPath))
        sys.exit()
    #subunit2
    try:
        cmd.create('tmp',"chain {}".format(",".join(subunit2)))
        cmd.save(os.path.join(savePath,"{}_{}.pdb".format(pdbName,subunit2)),'tmp')
        cmd.delete("tmp")
    except:
        logging.error("chain {} error in interaction Info {} for pdb {}".format(subunit2,interactionInfo,pdbPath))
        sys.exit()
    cmd.delete('all')
    return


#从pdb生成对应的dssp
def getDSSP(pdbFile):
    if os.path.exists(pdbFile):
        p=PDBParser(QUIET=True)
        structure=p.get_structure("tmp",pdbFile)
        model=structure[0]
        dssp=DSSP(model,pdbFile,dssp='mkdssp')
        return dssp
    else:
        logging.error("no such pdb:{}".format(pdbFile))
        sys.exit()
    
#从dssp中获取acc
def getAccFromDSSP(dssp,mutation):
    chain=mutation[1]
    wtRes=mutation[0]
    muRes=mutation[-1]
    mutationSite=int(mutation[2:-1])
    try:
        siteDssp=dssp[(chain,mutationSite)]
        if siteDssp[1]!=wtRes:  #mutation中信息和结构中信息不匹配？
            logging.error("res not match in {} and {}".format(str(mutation),str(siteDssp)))
            sys.exit()
    except:
        logging.error("unknown error in dssp")
        sys.exit()
    ss=siteDssp[2]
    acc=float(siteDssp[3])
    return acc

#获取complex和突变链的突变位置野生型的rSA
def getACCDict(pdbFile,interactionInfo,outDir,allMutation):
    pdbName=os.path.basename(os.path.splitext(pdbFile)[0])
    mutation2ACC={}
    splitPDB2Part(pdbFile,outDir,interactionInfo) #pymol切分complex得到subunit结构，得到E链和I链，以PDB格式存储
    complexDSSP=getDSSP(pdbFile)
    subUnit=interactionInfo.split("_")
    subUnit2DSSP={}
    for each in subUnit:
        subUnit2DSSP[each]=getDSSP(os.path.join(outDir,pdbName+"_"+each+".pdb"))
    for mutation in allMutation:
        #complex acc
        complexACC=getAccFromDSSP(complexDSSP,mutation)
        #chain=mutation[1]
        if mutation[1] in subUnit[0]:
            chain=subUnit[0]
        else:
            chain=subUnit[1]
        chainACC=getAccFromDSSP(subUnit2DSSP[chain],mutation)
        mutation2ACC[mutation]=[complexACC,chainACC]  #float类型
    return mutation2ACC