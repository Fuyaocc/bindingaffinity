#从结构中获取每条链的所有残基和interface上的残基
#不是看chain之间的interface，而是interact的chain group之间的interface
#顺便从结构中读取序列出来
import Bio
from Bio.PDB.PDBParser import PDBParser
import numpy as np
import logging
import sys
import os
def getInterfaceRateFromInterfaceRes(pdbName,interactionInfo,allRes,interfaceRes):
    subUnit=interactionInfo.split("_")
    chain2InterfaceRate={}
    for eachSubUnit in subUnit:
        chain2InterfaceRate[pdbName+"_"+eachSubUnit]=len(interfaceRes[eachSubUnit])/sum([len(allRes[each]) for each in eachSubUnit])
    return chain2InterfaceRate

def getInterfaceRateAndSeq(pdbPath,interfaceDis=8):
    #pdbName
    pdbName=os.path.basename(os.path.splitext(pdbPath)[0])
    chainGroup=[]
    parser = PDBParser()
    structure = parser.get_structure("temp", pdbPath)
    interactionInfo=''
    for chain in structure.get_chains():
        chainGroup.append(chain.get_id())
        interactionInfo=interactionInfo+'_'+chain.get_id()
    # print(chainGroup)
    interactionInfo=interactionInfo[1:]
    # print(interactionInfo)
    #先计算interface residue
    if pdbName=='inl0':
        chainGroup=['L','H']
        interactionInfo='L_H'
    if len(chainGroup)!=2:
        return -1,0,0
    
    model=structure[0]
    allRes={}  #complex每条链的有效残基
    complexSequence={} #complex中每条链的序列
    CAResName=[]  #残基名称，如E_S38
    CACoor=[] #残基对应的CA坐标
    chainID_in_PDB=set()#无序不重复集
    #提取所有的坐标
    for chain in model:
        minIndex=9999
        maxIndex=-1 #记录序列的起始位置
        chainID=chain.get_id()
        if chainID==" ":  #有些链是空的
            continue
        allRes[chainID]=set()#空集
        complexSequence[pdbName+'_'+chainID]=list("X"*2048)  #初始化为全为X，序列长为1024的列表
        chainID_in_PDB.add(chainID)
        for res in chain.get_residues():#得到有效残基allRes，序列complexSequence,残基名称及坐标CAResName&CACoor
            resID=res.get_id()[1]
            resName=res.get_resname()
            # print(str(resID)+' '+resName+' '+res.get_id()[0])
            if res.get_id()[0]!=" ":   # 非残基，一般为HOH
                continue
            try:
                if resName == "UNK":#UNK 未知
                    resName = "X"
                else:
                    resName = Bio.PDB.Polypeptide.three_to_one(resName)
            except KeyError:  #不正常的resName
                continue
            try:
                resCoor=res["CA"].get_coord()
            except KeyError:
                continue
            complexSequence[pdbName+'_'+chainID][resID-1]=resName
            if minIndex>resID:
                minIndex=resID
            if maxIndex<resID:
                maxIndex=resID
            allRes[chainID].add(resName+str(resID))
            resCoor=res["CA"].get_coord()
            CAResName.append(chainID+"_"+resName+str(resID))
            CACoor.append(resCoor)
        complexSequence[pdbName+'_'+chainID]=complexSequence[pdbName+'_'+chainID][minIndex-1:maxIndex]#截取残基链
        complexSequence[pdbName+'_'+chainID]=["".join(complexSequence[pdbName+'_'+chainID]),minIndex] #序列信息和序列起始位置
    #判断PDB中的链和interaction info中的链是否完全一样
    chainID_in_interactionInfo=set(interactionInfo)
    chainID_in_interactionInfo.remove("_")
    if not chainID_in_PDB==chainID_in_interactionInfo:
        logging.error("chain in PDB: {}, chain in interaction info {}, not match!".
                      format(str(chainID_in_PDB),str(chainID_in_interactionInfo)))
        #sys.exit()
    #计算distance map
    CACoor=np.array(CACoor)
    dis =  np.linalg.norm(CACoor[:, None, :] - CACoor[None, :, :], axis=-1)
    mask = dis<=interfaceDis
    resNumber=len(CAResName)
    #统计interface residue数量
    interfaceRes={}
    interfaceRes[chainGroup[0]]=set()
    interfaceRes[chainGroup[1]]=set()
    for i in range(resNumber):
        for j in range(i+1,resNumber):
            if mask[i][j]:
                #两条链分属于不同的chain group
                if CAResName[i][0] in chainGroup[0] and CAResName[j][0] in chainGroup[1]:
                    interfaceRes[chainGroup[0]].add(CAResName[i])
                    interfaceRes[chainGroup[1]].add(CAResName[j])
                    continue
                if CAResName[i][0] in chainGroup[1] and CAResName[j][0] in chainGroup[0]:
                    interfaceRes[chainGroup[1]].add(CAResName[i])
                    interfaceRes[chainGroup[0]].add(CAResName[j])
                    continue
    # interfaceRateDict=getInterfaceRateFromInterfaceRes(pdbName,interactionInfo,allRes,interfaceRes)
    return complexSequence,interfaceRes,chainGroup

if __name__ == '__main__':
    seq,interfaceDict=getInterfaceRateAndSeq('../data/1ay7.pdb','A_B')
    print(seq)
    print(interfaceDict)
