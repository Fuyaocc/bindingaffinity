import re
seqdic={}
affinitydic={}
# for line in open('../data/pdbbind_seq.txt'):
#     data=re.split('\t|\n',line)
#     seqdic[data[0]]=data[1]

for line in open('../data/benchmark_seq.txt'):
    data=re.split('\t|\n',line)
    if data[0] not in seqdic.keys():
        seqdic[data[0]]=data[1]

# for line in open('../data/pdbbind_affinity.txt'):
#     data=re.split('\t|\n',line)
#     pdbname=data[0].split('_')[0]
#     affinitydic[pdbname]=data[2]

for line in open('../data/benchmark_affinity.txt'):
    data=re.split('\t|\n',line)
    pdbname=data[0].split('_')[0]
    if pdbname not in affinitydic.keys():
        affinitydic[pdbname]=data[2]

seqfile=open('../data/benchmark_seq.txt','w+')
affinityfile=open('../data/benchmark_affinity.txt','w+')

for key in seqdic.keys():
    seqfile.write(key)
    seqfile.write('\t')
    seqfile.write(seqdic[key])
    seqfile.write('\n')

for key in affinitydic.keys():
    affinityfile.write(key)
    affinityfile.write('\t')
    affinityfile.write(affinitydic[key])
    affinityfile.write('\n')