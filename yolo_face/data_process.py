import numpy as np
import os
import cv2
import random
import shutil

def get_img_list(imgdir, listpath):
    list_file = open(listpath, 'w+')
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            if file.endswith("txt"):
                continue
            else:
                root = root.replace('\\', '/')
                imgpath = root + "/" + file
                list_file.write(imgpath + "\n")
    list_file.close()

def shuffle_txt(srctxt, shuffletxt):
    FileNamelist = []
    files = open(srctxt, 'r+')
    for line in files:
        line = line.strip('\n')  # 删除每一行的\n
        FileNamelist.append(line)
    print('len ( FileNamelist ) = ', len(FileNamelist))
    files.close()
    random.shuffle(FileNamelist)

    file_handle = open(shuffletxt, mode='w+')
    for idx in range(len(FileNamelist)):
        str = FileNamelist[idx]
        file_handle.write(str)
        file_handle.write("\n")
    file_handle.close()


if __name__ == "__main__":
    imgd = "D:/data/imgs/widerface_clean/train"
    txtlist = "D:/data/imgs/widerface_clean/train.txt"
    shufflepath = "D:/data/imgs/widerface_clean/train_shuffle.txt"
    get_img_list(imgd, txtlist)
    shuffle_txt(txtlist, shufflepath)


















