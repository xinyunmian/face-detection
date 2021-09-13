import torch
import os
import random
import numpy as np
import cv2
import math
from ParseYamlModel import Yaml2Pytorch
from FaceConfig import facecfg
from FaceUtils import *
device = "cpu"

def img_process(img, cfg):
    """将输入图片转换成网络需要的tensor
            Args:
                img_path: 人脸图片路径
            Returns:
                tensor： img(batch, channel, width, height)
    """
    h0, w0 = img.shape[:2]  # orig hw
    img_size = max(h0, w0)
    scalw = 1.0
    scalh = 1.0

    if cfg.padding_img:
        if cfg.input_size > 0:
            img_size = cfg.input_size
        r = img_size / max(h0, w0)
        im = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
        divisor = np.array(cfg.strides).max()
        imgsz = math.ceil(img_size / divisor) * divisor
        im = letterbox(im, new_shape=imgsz)
    else:
        res_scal = cfg.input_size / img_size
        neww = (int(w0 * res_scal / 32) + 1) * 32
        newh = (int(h0 * res_scal / 32) + 1) * 32
        im = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_LINEAR)
        scalw = neww / w0
        scalh = newh / h0

    im = im.astype(np.float32)
    im = im / 255.0
    im = im.transpose(2, 0, 1).copy()
    im = torch.from_numpy(im)
    im = im.unsqueeze(0)
    im = im.to(device)
    return im, scalw, scalh

def show_results(img, xywh, conf, landmarks, normxy=False):
    h, w, c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] - 0.5 * xywh[2])
    y1 = int(xywh[1] - 0.5 * xywh[3])
    x2 = int(xywh[0] + 0.5 * xywh[2])
    y2 = int(xywh[1] + 0.5 * xywh[3])

    if normxy:
        x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
        y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
        x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
        y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

    for i in range(5):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])

        if normxy:
            point_x = int(landmarks[2 * i] * w)
            point_y = int(landmarks[2 * i + 1] * h)

        cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:3]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def detect_one_img(faceNet, img_data, norm=False, noyaml=True, ylancors=[]):
    orgimg = img_data.copy()
    img_data, ws, hs = img_process(img=img_data, cfg=facecfg)
    outps = faceNet(img_data)[1]
    pocess_out = YoloFaceOutputProcess(outps, facecfg, not_yaml=noyaml, yaml_ancors=ylancors)
    pred = YoloFaceNMS(pocess_out, facecfg.conf_thresh, facecfg.nms_thresh)
    for i, det in enumerate(pred):
        orgimgwhc = np.array(orgimg.shape)

        gn_bx = torch.tensor(orgimgwhc)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
        gn_lks = torch.tensor(orgimgwhc)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks

        faceNum = det.size()[0]
        if faceNum > 0:
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords_boxes(img_data.shape[2:], det[:, :4], orgimgwhc).round()
            det[:, 5:15] = scale_coords_landmarks(img_data.shape[2:], det[:, 5:15], orgimgwhc).round()

            for j in range(faceNum):
                xywh = xyxy2xywh(det[j, :4].view(1, 4)).view(-1).tolist()
                conf = det[j, 4].item()
                class_num = det[j, 15].item()
                # class_num = det[j, 15].cpu().numpy()
                # conf = det[j, 4].cpu().numpy()
                landmarks = det[j, 5:15].view(-1).tolist()


                if norm:
                    xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn_bx).view(-1).tolist()
                    landmarks = (det[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()

                orgimg = show_results(orgimg, xywh, conf, landmarks, normxy=norm)
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.imshow('result', orgimg)
    cv2.waitKey(0)

def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}

if __name__ == "__main__":
    use_pretrain = True

    dnet = Yaml2Pytorch(cfg=facecfg.yaml_model, ch=3, nc=1)  # 需要修改
    anchors = dnet.yaml["anchors"]
    anchors = np.array(anchors, np.float32).reshape([3, 3, 2])

    if use_pretrain:
        d_path = "weights/yolov5n-0.5.pth"
        d_dict = torch.load(d_path, map_location=lambda storage, loc: storage)
        dnet.load_state_dict(d_dict, strict=False)
    else:
        d_path = "weights/FaceShuffle_150.pth"  # 需要修改
        d_dict = torch.load(d_path, map_location=lambda storage, loc: storage)
        dnet.load_state_dict(d_dict, strict=False)

    dnet.eval()
    dnet = dnet.to(device)

    # D:/data/imgs/widerface_clean/images/resize/si0.jpg
    # D:/codes/pytorch_projects/yolo_face/imgs
    img = cv2.imread("D:/codes/pytorch_projects/yolo_face/imgs/test.jpg")
    detect_one_img(dnet, img, noyaml=False, ylancors=anchors)












