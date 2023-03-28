import os

if __name__ == '__main__':
    path1='/home/ysbgs/xky/PP'
    path2='/home/ysbgs/xky/cleandata'
    files= os.listdir(path1)
    flag=False
    for file in files:
        if not os.path.isdir(file): 
            infp=path1+'/'+file
            outfp=path2+'/'+file.split('.')[0]+'.pdb'
            with open(infp, "r") as inputFile,open(outfp,"w") as outFile:
                for line in inputFile:
                    if line.startswith('MODEL') and line[12:14]!=' 1':
                        flag=True
                        continue
                    elif line.startswith("ENDMDL"): 
                        flag=False
                        continue
                    if not line.startswith("HETATM") and not line.startswith("ANISOU") and flag==False:
                        outFile.write(line)


