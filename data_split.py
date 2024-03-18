import os
import random
from config import cfg

# 如何写一个属于自己的数据集，读取大数据集 或者你自己的数据集
# 就需要生成一个text文件 train val .text
class data_split():
    def __init__(self):
        self.source_path='/home/xcj/code/cc/NLT-master/data_split/'
        self.dataset=cfg.target_dataset
        self.mode=cfg.train_mode
        self.ratio=cfg.ratio
        self.files_path='/home/xcj/code/cc/ProcessedData/'+self.dataset+'/train/img'
        assert os.path.exists(self.files_path), "path: '{}' does not exist.".format(self.files_path)
        self.val=cfg.val
        self.train_savepath=self.source_path+self.dataset+'_scenes/'+self.mode+'/'+self.mode+'.txt'
        self.val_savepath=self.source_path+self.dataset+'_scenes/val/val.txt'
        if os.path.exists(self.train_savepath):
            os.remove(self.train_savepath)
        if os.path.exists(self.val_savepath):
            os.remove(self.val_savepath)
    def generate_split(self):

        random.seed(0)
        files_name = sorted([int(file.split(".")[0]) for file in os.listdir(self.files_path)])

        #sorted([file.split(".")[0] for file in os.listdir(self.files_path)])
        files_num = len(files_name)
        train_num=self.ratio*files_num

        val_index = random.sample(range(1, files_num+1), k=int(files_num * self.val))
        train_index = list(filter(lambda item: item not in val_index, files_name))
        train_index = random.sample(train_index,k=int(train_num))
        train_files = []
        val_files = []

        for index, file_name in enumerate(files_name):
            if index+1 in val_index:
                val_files.append(str(file_name))  # 如果在val的范围里面，就放到val中
            elif index+1 in train_index:
                train_files.append(str(file_name))

            # 建立两个TXT文件
        train_f = open(self.train_savepath, "x")
        eval_f = open(self.val_savepath, "x")
        # 通过一个换行符再通过.joint的方法将list列表拼接成一个字符
        train_f.write("\n".join(train_files))
        eval_f.write("\n".join(val_files))

if __name__ == "__main__":
    ds=data_split()
    ds.generate_split()
