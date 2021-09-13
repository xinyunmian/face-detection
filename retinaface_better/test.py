from __future__ import print_function
import os
import torch
import cv2
import numpy as np
import shutil

from create_anchors import PriorBox
from config import cfg_slimNet3 as cfg
from slim_net import FaceDetectSlimNet
from retinaface_utils import decode, decode_landm
from nms import py_cpu_nms

from save_dpcore_weights import pytorch_to_dpcoreParams, save_feature_channel

import time
import random
device = torch.device("cuda")

class test_imgfpn:
    def __init__(self, faceNet):
        super(test_imgfpn, self).__init__()
        self.rgb_mean = (104, 117, 123)  # bgr order
        self.std_mean = (58, 57, 59)
        self.net = faceNet
        self.conf_thresh = 0.5
        self.nms_thresh = 0.25
    def detect_one(self, imgpath, minface):
        img = cv2.imread(imgpath)
        img = np.float32(img)
        res_scal = 20 / float(minface)

        img = cv2.resize(img, None, None, fx=res_scal, fy=res_scal, interpolation=cv2.INTER_CUBIC)

        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        scale = scale.to(device)

        # 减去均值转成numpy
        im_height, im_width, _ = img.shape
        img -= self.rgb_mean
        img /= self.std_mean
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)

        # b, c, h, w = img.shape
        # save_feature_channel("txt/imgp.txt", img, b, c, h, w)

        loc, conf, landms = self.net(img)  # forward pass
        pointb = torch.full_like(loc, 0)
        pointx = landms[:, :, [0, 2, 4, 6]]
        pointy = landms[:, :, [1, 3, 5, 7]]
        maxx, maxix = torch.max(pointx[0, :, :], 1)
        minx, minix = torch.min(pointx[0, :, :], 1)
        maxy, maxiy = torch.max(pointy[0, :, :], 1)
        miny, miniy = torch.min(pointy[0, :, :], 1)
        boxw = maxx - minx
        boxh = maxy - miny
        pointb[:, :, 0] = minx
        pointb[:, :, 1] = miny
        pointb[:, :, 2] = boxw
        pointb[:, :, 3] = boxh

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / res_scal
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / res_scal
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.conf_thresh)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:5000]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        scoresadd = scores[:, np.newaxis]
        return boxes, scoresadd, landms

    def test_Pyramid(self, imgp, lab=False):
        imgsa = cv2.imread(imgp)
        face_box20, face_score20, landms20 = self.detect_one(imgp, 20)
        face_box40, face_score40, landms40 = self.detect_one(imgp, 40)
        face_box60, face_score60, landms60 = self.detect_one(imgp, 60)
        face_box80, face_score80, landms80 = self.detect_one(imgp, 80)
        face_box100, face_score100, landms100 = self.detect_one(imgp, 100)
        face_box120, face_score120, landms120 = self.detect_one(imgp, 120)
        boxes = np.vstack((face_box20, face_box40, face_box60, face_box80, face_box100, face_box120)).astype(np.float32, copy=False)
        scores = np.vstack((face_score20, face_score40, face_score60, face_score80, face_score100, face_score120)).astype(np.float32, copy=False)
        landms = np.vstack((landms20, landms40, landms60, landms80, landms100, landms120)).astype(np.float32, copy=False)
        dets = np.hstack((boxes, scores)).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_thresh)
        face_rect = dets[keep, :]
        key_points = landms[keep]
        if lab:
            return face_rect, key_points
        else:
            have_face = 0
            for box, lands in zip(face_rect, key_points):
                have_face += 1
                bx1 = box[0]
                by1 = box[1]
                bx2 = box[2]
                by2 = box[3]
                cv2.rectangle(imgsa, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            return imgsa, have_face

def detect_one_img(faceNet, img_data, minface):
    conf_thresh = 0.5
    nms_thresh = 0.3
    im_shape = img_data.shape
    im_size_max = np.max(im_shape[0:2])
    res_scal = 640 / im_size_max
    # res_scal = 20 / float(minface)
    neww = (int(im_shape[1] * res_scal / 64) + 1) * 64
    newh = (int(im_shape[0] * res_scal / 64) + 1) * 64
    scalw = neww / im_shape[1]
    scalh = newh / im_shape[0]

    img = np.float32(img_data)
    # img = cv2.resize(img, None, None, fx=res_scal, fy=res_scal, interpolation=cv2.INTER_CUBIC)
    img = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_LINEAR)
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    scale = scale.to(device)

    # 减去均值转成numpy
    im_height, im_width, _ = img.shape
    img /= 255.0
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)

    # b, c, h, w = img.shape
    # save_feature_channel("txt/imgp.txt", img, b, c, h, w)

    loc, conf, landms = faceNet(img)  # forward pass

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])

    # boxes = boxes * scale / res_scal

    boxes = boxes * scale
    boxes[:, (0, 2)] = boxes[:, (0, 2)] / scalw
    boxes[:, (1, 3)] = boxes[:, (1, 3)] / scalh

    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)

    # landms = landms * scale1 / res_scal

    landms = landms * scale1
    landms[:, (0, 2, 4, 6, 8)] = landms[:, (0, 2, 4, 6, 8)] / scalw
    landms[:, (1, 3, 5, 7, 9)] = landms[:, (1, 3, 5, 7, 9)] / scalh

    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > conf_thresh)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:5000]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_thresh)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    return dets, landms

def test_one(imgp, net, minface, dir=False):
    img_mat = cv2.imread(imgp, cv2.IMREAD_COLOR)
    have_face = 0
    if img_mat is not None:
        im_h, im_w, _ = img_mat.shape
        face_rect, key_points = detect_one_img(net, img_mat, minface)
        for box, lands in zip(face_rect, key_points):
            x = int(box[0])
            y = int(box[1])
            w = int(box[2] - box[0])
            h = int(box[3] - box[1])
            have_face += 1
            bx1 = box[0]
            by1 = box[1]
            bx2 = box[2]
            by2 = box[3]
            cv2.rectangle(img_mat, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            # pp = np.zeros(8, dtype=np.int32)
            # pp[0] = lands[0]
            # pp[1] = lands[1]
            # pp[2] = lands[2]
            # pp[3] = lands[3]
            # pp[4] = lands[6]
            # pp[5] = lands[7]
            # pp[6] = lands[8]
            # pp[7] = lands[9]

            # cv2.circle(img_mat, (lands[0], lands[1]), 1, (0, 0, 255), 4)
            # cv2.circle(img_mat, (lands[2], lands[3]), 1, (0, 255, 255), 4)
            # cv2.circle(img_mat, (lands[4], lands[5]), 1, (255, 0, 255), 4)
            # cv2.circle(img_mat, (lands[6], lands[7]), 1, (0, 255, 0), 4)
            # cv2.circle(img_mat, (lands[8], lands[9]), 1, (255, 0, 0), 4)
    if dir:
        return img_mat, have_face
    else:
        cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        cv2.imshow('result', img_mat)
        cv2.waitKey(0)

def test_dir(imgdir, savedir, net, minface):
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    errnum = 0
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            root = root.replace("\\", "/")
            rootsplit = root.split("/")
            zidir = rootsplit[-1]
            imgpath = root + "/" + file
            saveimg, facenum = test_one(imgpath, net, minface, dir=True)
            if saveimg is not None:
                cv2.imshow('result', saveimg)
                cv2.waitKey(100)

                # savepath = savedir + "/" + file
                # cv2.imwrite(savepath, saveimg)

                if facenum > 0:
                    errnum += 1
                    # savepath = savedir + "/" + zidir + "_" + file
                    savepath = savedir + "/" + file
                    # imgsv = cv2.imread(imgpath)
                    # cv2.imwrite(savepath, saveimg)
                else:
                    savepath = savedir + "/noface/" + file
                    # cv2.imwrite(savepath, saveimg)
    print("error detect image: {}".format(errnum))

def save_box_landmarks(imgdir, txtdir, net, minface):
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            imgname = file.split(".")[0]
            imgpath = imgdir + "/" + file
            savepath = "D:/data/imgs/facePicture/facepic/20210129/addface13/eee" + "/" + file
            txtpath = txtdir + "/" + imgname + ".txt"
            txtfile = open(txtpath, mode="w+")
            path_head = "# addbg2/" + file
            txtfile.write(path_head + "\n")

            img_mat = cv2.imread(imgpath, cv2.IMREAD_COLOR)
            im_h, im_w, _ = img_mat.shape
            face_rect, key_points = detect_one_img(net, img_mat, minface)
            facenum = 0
            for box, lands in zip(face_rect, key_points):
                facenum += 1
                x = int(box[0])
                y = int(box[1])
                w = int(box[2] - box[0])
                h = int(box[3] - box[1])
                fx = ('%d' % x)
                fy = ('%d' % y)
                fw = ('%d' % w)
                fh = ('%d' % h)
                lex = ('%d' % lands[0])
                ley = ('%d' % lands[1])
                rex = ('%d' % lands[2])
                rey = ('%d' % lands[3])
                nex = ('%d' % lands[4])
                ney = ('%d' % lands[5])
                lmx = ('%d' % lands[6])
                lmy = ('%d' % lands[7])
                rmx = ('%d' % lands[8])
                rmy = ('%d' % lands[9])
                # face_pos = fx + " " + fy + " " + fw + " " + fh + " " + lex + " " + ley + " 0 " + rex + " " + rey + " 0 " + nex + " " + ney + " 0 " + lmx + " " + lmy + " 0 " + rmx + " " + rmy + " 0 0"
                face_pos = fx + " " + fy + " " + fw + " " + fh + " -1 -1 0 -1 -1 0 -1 -1 0 -1 -1 0 -1 -1 0 0"
                txtfile.write(face_pos + "\n")

                cv2.rectangle(img_mat, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(img_mat, (lands[0], lands[1]), 1, (0, 0, 255), 4)
                cv2.circle(img_mat, (lands[2], lands[3]), 1, (0, 255, 255), 4)
                cv2.circle(img_mat, (lands[4], lands[5]), 1, (255, 0, 255), 4)
                cv2.circle(img_mat, (lands[6], lands[7]), 1, (0, 255, 0), 4)
                cv2.circle(img_mat, (lands[8], lands[9]), 1, (255, 0, 0), 4)

            cv2.imshow('result', img_mat)
            cv2.waitKey(1)
            if facenum > 0:
                shutil.move(imgpath, savepath)
                # img_s = cv2.imread(imgpath, cv2.IMREAD_COLOR)
                # cv2.imwrite(savepath, img_mat)
            txtfile.close()

def get_crop_box(rect, imgw, imgh):
    bx = rect[0]
    by = rect[1]
    bw = rect[2] - rect[0]
    bh = rect[3] - rect[1]

    # #rings-laces
    # nbx1 = bx - 0.7 * bw #0.4
    # nby1 = by - 0.4 * bh #0.28
    # nbx2 = nbx1 + 2.4 * bw #1.8
    # nby2 = nby1 + 2.4 * bh #1.7

    # pose-shouder
    nbx1 = bx - 1.15 * bw  # 0.4
    nby1 = by - 0.28 * bh  # 0.28
    nbx2 = nbx1 + 3.3 * bw  # 1.8
    nby2 = nby1 + 2.6 * bh  # 1.7

    pp = np.zeros(4, dtype=np.int32)
    rx1 = max(nbx1, 0)
    ry1 = max(nby1, 0)
    rx2 = min(nbx2, imgw)
    ry2 = min(nby2, imgh)

    pp[0] = rx1
    pp[1] = ry1
    pp[2] = rx2
    pp[3] = ry2
    return pp

def show_crop_area(imgpath, dnet, minface):
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    imgdata = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    imh, imw, _ = imgdata.shape
    face_rect, key_points = detect_one_img(dnet, imgdata, minface)
    ii = 0
    for box, lands in zip(face_rect, key_points):
        rect = get_crop_box(box, imw, imh)
        cv2.rectangle(imgdata, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 4)
    cv2.imshow('result', imgdata)
    cv2.waitKey(0)

def crop_wanted_area(imgdir, savedir, dnet, minface):
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    id = 0
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            if file.endswith("txt"):
                continue
            else:
                filename = file.split(".")[0]
                imgpath = imgdir + "/" + file
                imgdata = cv2.imread(imgpath, cv2.IMREAD_COLOR)
                if imgdata is None:
                    print(imgpath)
                show_rot = imgdata
                imh, imw, _ = imgdata.shape
                face_rect, key_points = detect_one_img(dnet, imgdata, minface)
                ii = 0
                for box, lands in zip(face_rect, key_points):
                    rect = get_crop_box(box, imw, imh)
                    roi = imgdata[rect[1]:rect[3], rect[0]:rect[2], :]

                    # resize
                    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
                    interp_method = interp_methods[random.randrange(5)]
                    roi = cv2.resize(roi, (256, 256), interpolation=interp_method)

                    savename = str(ii) + file
                    savepath = savedir + "/" + file
                    ii += 1
                    # pp = np.zeros(4, dtype=np.int32)
                    # pp[0] = box[0]
                    # pp[1] = box[1]
                    # pp[2] = box[2]
                    # pp[3] = box[3]
                    # roi = imgdata[pp[1]:pp[3], pp[0]:pp[2], :]
                    # savename = "error_" + str(id) + ".jpg"
                    # savepath = savedir + "/" + savename
                    # id += 1

                    cv2.imwrite(savepath, roi)
                    cv2.rectangle(show_rot, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
            cv2.imshow('result', show_rot)
            cv2.waitKey(1)

def rename_to_jpg(imgdir, savedir):
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            filename = file.split(".")[0]
            imgpath = imgdir + "/" + file
            imgdata = cv2.imread(imgpath, cv2.IMREAD_COLOR)
            savename = filename + ".jpg"
            savepath = savedir + "/" + savename
            cv2.imwrite(savepath, imgdata)

def test_fpndir(net, imgdir, savedir):
    FPNTest = test_imgfpn(net)
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            root = root.replace("\\", "/")
            rootsplit = root.split("/")
            zidir = rootsplit[-1]
            imgpath = root + "/" + file
            saveimg, facenum= FPNTest.test_Pyramid(imgpath)
            if saveimg is not None:
                cv2.imshow('result', saveimg)
                cv2.waitKey(1)
                if facenum > 0:
                    savepath = savedir + "/" + zidir + "/" + file
                    cv2.imwrite(savepath, saveimg)

def save_label_fpn(imgdir, txtdir, saveresult, net, subdirs=False):
    FPNTest = test_imgfpn(net)
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            if subdirs:
                root = root.replace("\\", "/")
                rootsplit = root.split("/")
                zidir = rootsplit[-1]
                imgname = file.split(".")[0]
                imgpath = root + "/" + file
                savepath = saveresult + "/" + zidir + "/" + file
                txtpath = txtdir + "/" + zidir + "/" + imgname + ".txt"
            else:
                imgname = file.split(".")[0]
                zidir = imgdir.split("/")[-1]
                imgpath = imgdir + "/" + file
                savepath = saveresult + "/" + file
                txtpath = txtdir + "/" + imgname + ".txt"

            txtfile = open(txtpath, mode="w+")
            path_head = "# " + zidir + "/" + file
            txtfile.write(path_head + "\n")

            img_mat = cv2.imread(imgpath, cv2.IMREAD_COLOR)
            im_h, im_w, _ = img_mat.shape
            face_rect, key_points = FPNTest.test_Pyramid(imgpath, lab=True)
            facenum = 0
            for box, lands in zip(face_rect, key_points):
                facenum += 1
                x = int(box[0])
                y = int(box[1])
                w = int(box[2] - box[0])
                h = int(box[3] - box[1])
                fx = ('%d' % x)
                fy = ('%d' % y)
                fw = ('%d' % w)
                fh = ('%d' % h)
                face_pos = fx + " " + fy + " " + fw + " " + fh + " -1 -1 0 -1 -1 0 -1 -1 0 -1 -1 0 -1 -1 0 1"
                txtfile.write(face_pos + "\n")

                cv2.rectangle(img_mat, (x, y), (x + w, y + h), (0, 255, 0), 4)

            cv2.imshow('result', img_mat)
            cv2.waitKey(1)
            # if facenum > 0:
            #     cv2.imwrite(savepath, img_mat)
            txtfile.close()

if __name__ == "__main__":
    dnet = FaceDetectSlimNet(cfg=cfg)  # 需要修改
    d_path = "weights/face_slim_0609_250.pth"  # 需要修改
    d_dict = torch.load(d_path, map_location=lambda storage, loc: storage)
    dnet.load_state_dict(d_dict)
    dnet.eval()
    dnet = dnet.to(device)
    # saveparams = pytorch_to_dpcoreParams(dnet)
    # saveparams.forward("facedet_param_cfg.h", "facedet_param_src.h")


    imgpath = r"D:\codes\pytorch_projects\yolov5-face\data\images/test.jpg"
    savepath = "D:/data/imgs/facePicture/ears/bb/crop_ear"
    # rename_to_jpg(imgpath, savepath)


    min_face = 20
    # FPNTest = test_imgfpn(dnet)
    # result = FPNTest.test_Pyramid(imgpath)
    # cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    # cv2.imshow('result', result)
    # cv2.waitKey(0)

    # test_one(imgpath, dnet, min_face, dir=False)

    img_dir = "D:/data/imgs/facePicture/pose_person/pose_shouder/app" #addface11/background
    txt_dir = "D:/data/imgs/facePicture/facepic/aatxt/faceeva"
    save_dir = "D:/data/imgs/facePicture/pose_person/pose_shouder/test4"
    # test_dir(img_dir, save_dir, dnet, min_face)
    # test_fpndir(dnet, img_dir, save_dir)
    # save_box_landmarks(img_dir, txt_dir, dnet, min_face)
    # save_label_fpn(img_dir, txt_dir, save_dir, dnet, subdirs=False)
    crop_wanted_area(img_dir, save_dir, dnet, min_face)

    showpath = "D:/data/imgs/facePicture/pose_person/pose/0/0_addp(108).png"
    # show_crop_area(showpath, dnet, min_face)


















