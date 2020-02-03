import os
import glob
# import codecs


path='../data'
files = glob.glob(path+'/*.txt')
for file in files:
    print(file)
    f = open(file,'r')
    filedata = f.read()
    f.close()
    newdata=filedata.replace('  ',' ').lower()
    f = open(file,'w')
    f.write(newdata)
    f.close()