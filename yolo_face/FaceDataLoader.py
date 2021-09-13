import torch
import torch.utils.data as data
import cv2
import numpy as np
from multiprocessing.pool import ThreadPool
import random
import math

from pathlib import Path
import os
import glob
from tqdm import tqdm
from PIL import Image, ExifTags
import logging
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class LoadFace(data.Dataset):
    def __init__(self, img_path, cfg):
        super(LoadFace, self).__init__()
        self.config = cfg
        self.mosaic_border = [-self.config.img_size // 2, -self.config.img_size // 2]
        self.labels = []
        self.img_files = []
        f = open(img_path, 'r', encoding='utf-8')
        for imgp in f.readlines():
            imgp = imgp.rstrip()
            img_name = imgp.split("/")[-1]
            img_hz = img_name.split(".")[-1]
            labelp = imgp.replace(img_hz, 'txt')
            labs = open(labelp, 'r')
            labline = labs.readlines()
            if len(labline) == 0:
                continue
            else:
                self.img_files.append(imgp)
                laboneimg = []
                for lab in labline:
                    lab = lab.strip().split(" ")
                    self.remove_string_list(lab, words="")
                    class_id = int(lab[0])
                    cx = float(lab[1])
                    cy = float(lab[2])
                    iw = float(lab[3])
                    ih = float(lab[4])

                    lex = float(lab[5])
                    ley = float(lab[6])
                    rex = float(lab[7])
                    rey = float(lab[8])
                    nox = float(lab[9])
                    noy = float(lab[10])
                    lmx = float(lab[11])
                    lmy = float(lab[12])
                    rmx = float(lab[13])
                    rmy = float(lab[14])
                    labone = [class_id, cx, cy, iw, ih, lex, ley, rex, rey, nox, noy, lmx, lmy, rmx, rmy]
                    laboneimg.extend(labone)
                labarry = np.array(laboneimg, dtype=np.float32)
                labarry = np.reshape(labarry, (-1, 15))
                self.labels.append(labarry)

        nimg = len(self.labels)
        self.nums = nimg
        self.indices = range(nimg)

    def remove_string_list(self, movelist, words=" "):
        for i in range(len(movelist) - 1):
            if movelist[i] == words:
                movelist.remove(movelist[i])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        index = self.indices[index]
        img_moasic = self.config.moasic
        if img_moasic:
            img, labels = self.load_mosaic_face(index)
            shapes = None
        else:
            # Load image
            img, (h0, w0), (h, w) = self.load_image(index)
            shape = self.config.img_size
            img, ratio, pad = self.letterbox(img, shape, auto=False, scaleup=True)
            shapes = (h0, w0), ((h / h0, w / w0), pad)

            # Load labels
            labels = []
            x = self.labels[index]
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]
                labels[:, 5] = np.array(x[:, 5] > 0, dtype=np.int32) * (ratio[0] * w * x[:, 5] + pad[0]) + (np.array(x[:, 5] > 0, dtype=np.int32) - 1)
                labels[:, 6] = np.array(x[:, 6] > 0, dtype=np.int32) * (ratio[1] * h * x[:, 6] + pad[1]) + (np.array(x[:, 6] > 0, dtype=np.int32) - 1)
                labels[:, 7] = np.array(x[:, 7] > 0, dtype=np.int32) * (ratio[0] * w * x[:, 7] + pad[0]) + (np.array(x[:, 7] > 0, dtype=np.int32) - 1)
                labels[:, 8] = np.array(x[:, 8] > 0, dtype=np.int32) * (ratio[1] * h * x[:, 8] + pad[1]) + (np.array(x[:, 8] > 0, dtype=np.int32) - 1)
                labels[:, 9] = np.array(x[:, 5] > 0, dtype=np.int32) * (ratio[0] * w * x[:, 9] + pad[0]) + (np.array(x[:, 9] > 0, dtype=np.int32) - 1)
                labels[:, 10] = np.array(x[:, 5] > 0, dtype=np.int32) * (ratio[1] * h * x[:, 10] + pad[1]) + (np.array(x[:, 10] > 0, dtype=np.int32) - 1)
                labels[:, 11] = np.array(x[:, 11] > 0, dtype=np.int32) * (ratio[0] * w * x[:, 11] + pad[0]) + ( np.array(x[:, 11] > 0, dtype=np.int32) - 1)
                labels[:, 12] = np.array(x[:, 12] > 0, dtype=np.int32) * (ratio[1] * h * x[:, 12] + pad[1]) + ( np.array(x[:, 12] > 0, dtype=np.int32) - 1)
                labels[:, 13] = np.array(x[:, 13] > 0, dtype=np.int32) * (ratio[0] * w * x[:, 13] + pad[0]) + (np.array(x[:, 13] > 0, dtype=np.int32) - 1)
                labels[:, 14] = np.array(x[:, 14] > 0, dtype=np.int32) * (ratio[1] * h * x[:, 14] + pad[1]) + (np.array(x[:, 14] > 0, dtype=np.int32) - 1)

        # 是否进行数据增强
        if self.config.augment:
            if not img_moasic:
                img, labels = self.random_perspective(img, labels)

            # Augment colorspace
            hsv_h = self.config.hsv_h
            hsv_s = self.config.hsv_s
            hsv_v = self.config.hsv_v
            self.augment_hsv(img, hgain=hsv_h, sgain=hsv_s, vgain=hsv_v)

        nL = len(labels)  # number of labels
        if nL:
            labels[:, 1:5] = self.xyxy2xywh(labels[:, 1:5])  # convert x1 y1 x2 y2 to cx cy w h
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

            labels[:, [5, 7, 9, 11, 13]] /= img.shape[1]  # normalized landmark x 0-1
            labels[:, [5, 7, 9, 11, 13]] = np.where(labels[:, [5, 7, 9, 11, 13]] < 0, -1, labels[:, [5, 7, 9, 11, 13]])
            labels[:, [6, 8, 10, 12, 14]] /= img.shape[0]  # normalized landmark y 0-1
            labels[:, [6, 8, 10, 12, 14]] = np.where(labels[:, [6, 8, 10, 12, 14]] < 0, -1, labels[:, [6, 8, 10, 12, 14]])

        if self.config.augment:
            # flip up-down
            if random.random() < self.config.flipud:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]
                    labels[:, 6] = np.where(labels[:, 6] < 0, -1, 1 - labels[:, 6])
                    labels[:, 8] = np.where(labels[:, 8] < 0, -1, 1 - labels[:, 8])
                    labels[:, 10] = np.where(labels[:, 10] < 0, -1, 1 - labels[:, 10])
                    labels[:, 12] = np.where(labels[:, 12] < 0, -1, 1 - labels[:, 12])
                    labels[:, 14] = np.where(labels[:, 14] < 0, -1, 1 - labels[:, 14])

            # flip left-right
            if random.random() < self.config.fliplr:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]
                    labels[:, 5] = np.where(labels[:, 5] < 0, -1, 1 - labels[:, 5])
                    labels[:, 7] = np.where(labels[:, 7] < 0, -1, 1 - labels[:, 7])
                    labels[:, 9] = np.where(labels[:, 9] < 0, -1, 1 - labels[:, 9])
                    labels[:, 11] = np.where(labels[:, 11] < 0, -1, 1 - labels[:, 11])
                    labels[:, 13] = np.where(labels[:, 13] < 0, -1, 1 - labels[:, 13])

                    # 左右镜像的时候，左眼、右眼，　左嘴角、右嘴角无法区分, 应该交换位置，便于网络学习
                    eye_left = np.copy(labels[:, [5, 6]])
                    mouth_left = np.copy(labels[:, [11, 12]])
                    labels[:, [5, 6]] = labels[:, [7, 8]]
                    labels[:, [7, 8]] = eye_left
                    labels[:, [11, 12]] = labels[:, [13, 14]]
                    labels[:, [13, 14]] = mouth_left

        labels_out = torch.zeros((nL, 16))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)
            self.showlabels(img.copy(), labels[:, 1:5], labels[:, 5:15])

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    def showlabels(self, img, boxs, landmarks):
        for box in boxs:
            x, y, w, h = box[0] * img.shape[1], box[1] * img.shape[0], box[2] * img.shape[1], box[3] * img.shape[0]
            # cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0,255,0), 2)
            cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)

        for landmark in landmarks:
            # cv2.circle(img,(60,60),30,(0,0,255))
            for i in range(5):
                cv2.circle(img, (int(landmark[2 * i] * img.shape[1]), int(landmark[2 * i + 1] * img.shape[0])), 3, (0, 0, 255), -1)
        cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        cv2.imshow('result', img)
        cv2.waitKey(0)

    def xyxy2xywh(self, x):
        # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y

    def augment_hsv(self, img, hgain=0.5, sgain=0.5, vgain=0.5):
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype  # uint8
        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def load_image(self, index):
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.config.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:
            interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
            interp_method = interp_methods[random.randrange(5)]
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp_method)
        return img, (h0, w0), img.shape[:2]

    def load_mosaic_face(self, index):
        labels4 = []
        s = self.config.img_size
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
        indices = [index] + [self.indices[random.randint(0, self.nums - 1)] for _ in range(3)]  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b
            # Labels
            x = self.labels[index]
            labels = x.copy()
            if x.size > 0:  # Normalized xywh to pixel xyxy format
                # box, x1,y1,x2,y2
                labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
                labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
                labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
                labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh

                # 10 landmarks
                labels[:, 5] = np.array(x[:, 5] > 0, dtype=np.int32) * (w * x[:, 5] + padw) + (np.array(x[:, 5] > 0, dtype=np.int32) - 1)
                labels[:, 6] = np.array(x[:, 6] > 0, dtype=np.int32) * (h * x[:, 6] + padh) + (np.array(x[:, 6] > 0, dtype=np.int32) - 1)
                labels[:, 7] = np.array(x[:, 7] > 0, dtype=np.int32) * (w * x[:, 7] + padw) + (np.array(x[:, 7] > 0, dtype=np.int32) - 1)
                labels[:, 8] = np.array(x[:, 8] > 0, dtype=np.int32) * (h * x[:, 8] + padh) + (np.array(x[:, 8] > 0, dtype=np.int32) - 1)
                labels[:, 9] = np.array(x[:, 9] > 0, dtype=np.int32) * (w * x[:, 9] + padw) + ( np.array(x[:, 9] > 0, dtype=np.int32) - 1)
                labels[:, 10] = np.array(x[:, 10] > 0, dtype=np.int32) * (h * x[:, 10] + padh) + (np.array(x[:, 10] > 0, dtype=np.int32) - 1)
                labels[:, 11] = np.array(x[:, 11] > 0, dtype=np.int32) * (w * x[:, 11] + padw) + (np.array(x[:, 11] > 0, dtype=np.int32) - 1)
                labels[:, 12] = np.array(x[:, 12] > 0, dtype=np.int32) * (h * x[:, 12] + padh) + (np.array(x[:, 12] > 0, dtype=np.int32) - 1)
                labels[:, 13] = np.array(x[:, 13] > 0, dtype=np.int32) * (w * x[:, 13] + padw) + (np.array(x[:, 13] > 0, dtype=np.int32) - 1)
                labels[:, 14] = np.array(x[:, 14] > 0, dtype=np.int32) * (h * x[:, 14] + padh) + (np.array(x[:, 14] > 0, dtype=np.int32) - 1)
            labels4.append(labels)
        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            np.clip(labels4[:, 1:5], 0, 2 * s, out=labels4[:, 1:5])  # use with random_perspective
            # landmarks
            labels4[:, 5:] = np.where(labels4[:, 5:] < 0, -1, labels4[:, 5:])
            labels4[:, 5:] = np.where(labels4[:, 5:] > 2 * s, -1, labels4[:, 5:])

            labels4[:, 5] = np.where(labels4[:, 6] == -1, -1, labels4[:, 5])
            labels4[:, 6] = np.where(labels4[:, 5] == -1, -1, labels4[:, 6])

            labels4[:, 7] = np.where(labels4[:, 8] == -1, -1, labels4[:, 7])
            labels4[:, 8] = np.where(labels4[:, 7] == -1, -1, labels4[:, 8])

            labels4[:, 9] = np.where(labels4[:, 10] == -1, -1, labels4[:, 9])
            labels4[:, 10] = np.where(labels4[:, 9] == -1, -1, labels4[:, 10])

            labels4[:, 11] = np.where(labels4[:, 12] == -1, -1, labels4[:, 11])
            labels4[:, 12] = np.where(labels4[:, 11] == -1, -1, labels4[:, 12])

            labels4[:, 13] = np.where(labels4[:, 14] == -1, -1, labels4[:, 13])
            labels4[:, 14] = np.where(labels4[:, 13] == -1, -1, labels4[:, 14])
        img4, labels4 = self.random_perspective(img4, labels4, border = self.mosaic_border)
        return img4, labels4

    def random_perspective(self, img, targets=(), border=(0, 0)):
        degrees = self.config.degrees
        translate = self.config.translate
        scale = self.config.scale
        shear = self.config.shear
        perspective = self.config.perspective
        height = img.shape[0] + border[0] * 2  # shape(h,w,c)
        width = img.shape[1] + border[1] * 2

        # Center
        C = np.eye(3)
        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-degrees, degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - scale, 1 + scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if perspective:
                img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

        n = len(targets)
        if n:
            xy = np.ones((n * 9, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]].reshape(n * 9, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            if perspective:
                xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 18)  # rescale
            else:  # affine
                xy = xy[:, :2].reshape(n, 18)

                # create new boxes
                x = xy[:, [0, 2, 4, 6]]
                y = xy[:, [1, 3, 5, 7]]

                landmarks = xy[:, [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]]
                mask = np.array(targets[:, 5:] > 0, dtype=np.int32)
                landmarks = landmarks * mask
                landmarks = landmarks + mask - 1

                landmarks = np.where(landmarks < 0, -1, landmarks)
                landmarks[:, [0, 2, 4, 6, 8]] = np.where(landmarks[:, [0, 2, 4, 6, 8]] > width, -1, landmarks[:, [0, 2, 4, 6, 8]])
                landmarks[:, [1, 3, 5, 7, 9]] = np.where(landmarks[:, [1, 3, 5, 7, 9]] > height, -1, landmarks[:, [1, 3, 5, 7, 9]])

                landmarks[:, 0] = np.where(landmarks[:, 1] == -1, -1, landmarks[:, 0])
                landmarks[:, 1] = np.where(landmarks[:, 0] == -1, -1, landmarks[:, 1])

                landmarks[:, 2] = np.where(landmarks[:, 3] == -1, -1, landmarks[:, 2])
                landmarks[:, 3] = np.where(landmarks[:, 2] == -1, -1, landmarks[:, 3])

                landmarks[:, 4] = np.where(landmarks[:, 5] == -1, -1, landmarks[:, 4])
                landmarks[:, 5] = np.where(landmarks[:, 4] == -1, -1, landmarks[:, 5])

                landmarks[:, 6] = np.where(landmarks[:, 7] == -1, -1, landmarks[:, 6])
                landmarks[:, 7] = np.where(landmarks[:, 6] == -1, -1, landmarks[:, 7])

                landmarks[:, 8] = np.where(landmarks[:, 9] == -1, -1, landmarks[:, 8])
                landmarks[:, 9] = np.where(landmarks[:, 8] == -1, -1, landmarks[:, 9])

                targets[:, 5:] = landmarks

                xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

                # clip boxes
                xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
                xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

                # filter candidates
                i = self.box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T)
                targets = targets[i]
                targets[:, 1:5] = xy[i]
            return img, targets

    def box_candidates(self, box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1):
        # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates

def collate_face(batch):
    img, label, path, shapes = zip(*batch)  # transposed
    for i, l in enumerate(label):
        l[:, 0] = i  # add target image index for build_targets()
    return torch.stack(img, 0), torch.cat(label, 0), path, shapes

for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [x.replace(sa, sb, 1).replace('.' + x.split('.')[-1], '.txt') for x in img_paths]

def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))

def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s

def cache_labels(path, img_files, label_files):
    help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
    x = {}  # dict
    nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, duplicate
    pbar = tqdm(zip(img_files, label_files), desc='Scanning images', total=len(img_files))
    for i, (im_file, lb_file) in enumerate(pbar):
        try:
            im = Image.open(im_file)
            im.verify()  # PIL verify
            shape = exif_size(im)  # image size
            assert (shape[0] > 9) & (shape[1] > 9), 'image size <10 pixels'

            if os.path.isfile(lb_file):
                nf += 1  # label found
                with open(lb_file, 'r') as f:
                    l = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
                if len(l):
                    assert l.shape[1] == 15, 'labels require 15 columns each'
                    assert (l >= -1).all(), 'negative labels'
                    assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                    assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                else:
                    ne += 1  # label empty
                    l = np.zeros((0, 15), dtype=np.float32)
            else:
                nm += 1  # label missing
                l = np.zeros((0, 15), dtype=np.float32)
            x[im_file] = [l, shape]
        except Exception as e:
            nc += 1
            print('WARNING: Ignoring corrupted image and/or label %s: %s' % (im_file, e))

        pbar.desc = "Scanning '{}' for images and labels {} found, {} missing, {} empty, {} corrupted"\
            .format(path.parent / path.stem, nf, nm, ne, nc)
    if nf == 0:
        print('WARNING: No labels found in {path}. See {}'.format(help_url))

    x['hash'] = get_hash(label_files + img_files)
    x['results'] = [nf, nm, ne, nc, i + 1]
    torch.save(x, str(path))  # save for next time
    logging.info("New cache created: {}".format(path))
    return x



if __name__ == "__main__":
    facelist = "D:/data/imgs/widerface_clean/face.txt"
    from FaceConfig import facecfg
    data_train = LoadFace(facelist, facecfg)
    train_loader = data.DataLoader(data_train, batch_size=facecfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate_face)
    for i, (imgs, targets, paths, _) in enumerate(train_loader):
        imgsii = imgs
        lab = targets
        lists = paths
        # targets2 = targets.repeat(na, 1, 1)
        na = 3
        nt = targets.shape[0]
        ai = torch.arange(na, device="cpu").float().view(na, 1).repeat(1, nt)
        ai22 = ai[:, :, None]
        targets22 = targets.repeat(na, 1, 1)
        targets = torch.cat((targets22, ai22), 2)
        print("error")







    img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']
    path = "D:/data/imgs/widerface_clean/train"
    f = []
    for p in path if isinstance(path, list) else [path]:
        p = Path(p)  # os-agnostic
        if p.is_dir():  # dir
            f += glob.glob(str(p / '**' / '*.*'), recursive=True)

    img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats])
    label_files = img2label_paths(img_files)

    cache_path = Path(label_files[0]).parent.with_suffix('.cache')  # cached labels
    if cache_path.is_file():
        cache = torch.load(cache_path)  # load
        if cache['hash'] != get_hash(label_files + img_files) or 'results' not in cache:  # changed
            cache = cache_labels(cache_path, img_files, label_files)  # re-cache
    else:
        cache = cache_labels(cache_path, img_files, label_files)  # cache

    [nf, nm, ne, nc, n] = cache.pop('results')
    cache.pop('hash')  # remove hash
    labels, shapes = zip(*cache.values())
    tlab = list(labels)
    tshape = np.array(shapes, dtype=np.float64)
    timgs = list(cache.keys())  # update
    tlab_files = img2label_paths(cache.keys())  # update
    for x in tlab:
        x[:, 0] = 0

    n = len(shapes)  # number of images
    tbi = np.floor(np.arange(n) / 16).astype(np.int)  # batch index
    tnb = tbi[-1] + 1  # number of batches
    tbatch = tbi  # batch index of image
    tn = n
    indices = range(n)
    rectanguarTraing = True
    if rectanguarTraing:
        s = tshape  # wh
        ar = s[:, 1] / s[:, 0]  # aspect ratio
        irect = ar.argsort()
        timg_files = [timgs[i] for i in irect]
        label_files = [tlab_files[i] for i in irect]
        labels = [tlab[i] for i in irect]
        shapes = s[irect]  # wh
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * tnb
        for i in range(tnb):
            ari = ar[tbi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        batch_shapes = np.ceil(np.array(shapes) * 640 / 32 + 0).astype(np.int) * 32
    imgs = [None] * n
    cache_images = True
    if cache_images:
        gb = 0  # Gigabytes of cached images
        img_hw0, img_hw = [None] * n, [None] * n
        results = ThreadPool(8).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))  # 8 threads
        pbar = tqdm(enumerate(results), total=n)
        for i, x in pbar:
            imgs[i], img_hw0[i], img_hw[i] = x  # img, hw_original, hw_resized = load_image(self, i)
            gb += imgs[i].nbytes
            pbar.desc = 'Caching images (%.1fGB)' % (gb / 1E9)































