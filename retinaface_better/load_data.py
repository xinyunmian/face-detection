import torch
import torch.utils.data as data
import cv2
import numpy as np

class FaceDataLoader(data.Dataset):
    def __init__(self, txt_path, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        f = open(txt_path,'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = txt_path.replace('train_resize.txt','images/') + path
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                if label[2] < 8 or label[3] < 8:
                    labels.append(label)
                if len(label) < 8:
                    print(path)

        self.words.append(labels)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index], cv2.IMREAD_COLOR)
        if img is None:
            print(self.imgs_path[index])
        height, width, _ = img.shape

        labels = self.words[index]
        annotations = np.zeros((0, 15))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            if len(label) < 8:
                print(self.imgs_path[index])
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

            # landmarks
            annotation[0, 4] = label[4]    # l0_x
            annotation[0, 5] = label[5]    # l0_y
            annotation[0, 6] = label[7]    # l1_x
            annotation[0, 7] = label[8]    # l1_y
            annotation[0, 8] = label[10]   # l2_x
            annotation[0, 9] = label[11]   # l2_y
            annotation[0, 10] = label[13]  # l3_x
            annotation[0, 11] = label[14]  # l3_y
            annotation[0, 12] = label[16]  # l4_x
            annotation[0, 13] = label[17]  # l4_y

            if label[19] == 0:
                annotation[0, 14] = -10
            else:
                if (annotation[0, 4] < 0):
                    annotation[0, 14] = -1
                else:
                    annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)

        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target

        # img_patches = []
        # lab_patches = []
        # if self.preproc is not None:
        #     for _ in range(2):
        #         imgp, targetp = self.preproc(img, target)
        #         imgp = torch.from_numpy(imgp)
        #         img_patches.append(imgp)
        #         lab_patches.append(targetp)
        # return img_patches, lab_patches

def Loader_collate(batch):
    targets = []
    imgs = []

    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)

    # for _, sample in enumerate(batch):
    #     for _, tup in enumerate(sample):
    #         for _, datup in enumerate(tup):
    #             if torch.is_tensor(datup):
    #                 imgs.append(datup)
    #             elif isinstance(datup, type(np.empty(0))):
    #                 annos = torch.from_numpy(datup).float()
    #                 targets.append(annos)
    #
    # return (torch.stack(imgs, 0), targets)

if __name__ == "__main__":
    import os
    import shutil
    path = "D:/data/imgs/widerface/train_new2/images/face.txt"
    jpg_or_txt = "D:/data/imgs/widerface/train_new2/images/mouth_mask3"
    save = "D:/data/imgs/widerface/train_new2/images/mouth_mask"
    with open(path, 'r') as file:
        label_files = file.readlines()
    # img_files = [path_lab.replace('.txt', '.jpg') for path_lab in label_files]
    #########将满足条件的图片移动到指定文件夹#####################
    for path_lab in label_files:
        path_img = path_lab.replace(".jpg", ".txt")
        path_lab = path_lab.rstrip("\n")
        path_img = path_img.rstrip("\n")
        txtpath, txtname = os.path.split(path_lab)
        imgpath, imgname = os.path.split(path_img)

        # jpg_or_txtp = os.path.join(jpg_or_txt, imgname)
        # savetxt = os.path.join(save, imgname)
        # shutil.copyfile(jpg_or_txtp, savetxt)

        jpg_or_txtp = os.path.join(jpg_or_txt, txtname)
        savetxt = os.path.join(save, txtname)
        shutil.move(jpg_or_txtp, savetxt)
            #shutil.copyfile(path_lab, savetxt)











































