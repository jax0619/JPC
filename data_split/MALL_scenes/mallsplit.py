import os
import pandas as pd
from glob import glob
import numpy as np
import shutil

path='/home/xcj/code/cc/ProcessedData/MALLs/'
den_path='/home/xcj/code/cc/ProcessedData/MALLs/den/'
img_path='/home/xcj/code/cc/ProcessedData/MALLs/img/'

txt=glob(os.path.join(path, '*.txt'))
den=glob(os.path.join(den_path, '*.csv'))
img=glob(os.path.join(img_path, '*.jpg'))

dbtype_list = os.listdir(path)
dbtype_list_copy=dbtype_list.copy()

def mycopyfile(srcfile,dstpath):                       # 复制函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.copy(srcfile, dstpath + fname)          # 复制文件
        print ("copy %s -> %s"%(srcfile, dstpath + fname))

for dbtype in dbtype_list:
    if os.path.isfile(os.path.join(path, dbtype)):
        dbtype_list_copy.remove(dbtype)

for split_path in txt:
    #mode=split_path.split("L/")[1].split(".")[0]
    mode_file=os.path.basename(split_path)
    mode_name = mode_file.split(".")[0]
    den_targetpath=path+mode_name+'/'+'den/'
    img_targetpath = path + mode_name + '/' + 'img/'
    filepath=path+mode_file
    file=pd.read_csv(filepath)
    filend=file.values
    for num in range(len(filend)):
        jpgpath=img_path+str(filend[num][0])+'.jpg'
        csvpath=den_path+str(filend[num][0])+'.csv'
        mycopyfile(jpgpath,img_targetpath)
        mycopyfile(csvpath,den_targetpath)

