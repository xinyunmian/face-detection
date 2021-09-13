import numpy as np

class face_config(object):
    #2.numeric parameters
    start_epoch = 0
    epochs = 501
    batch_size = 4
    img_size = 640

    # augment params
    augment = 1
    moasic = 0
    hsv_h = 0.015
    hsv_s = 0.7
    hsv_v = 0.4
    flipud = 0.0
    fliplr = 0.5
    degrees = 0.0  # image rotation (+/- deg)
    translate = 0.1  # image translation (+/- fraction)
    scale = 0.5  # image scale (+/- gain)
    shear = 0.5  # image shear (+/- deg)
    perspective = 0.0  # image perspective (+/- fraction), range 0-0.001
    anchor_box_ratio = 4.0

    # net params
    strides = [8, 16, 32]
    # out_channels = [16, 32, 64, 128, 256, 512]
    # target_outc =3 * (5 + 1 + 10)
    # fpn_in_list = [64, 128, 256, 512]
    # fpn_out_list = [64, 128, 256, 512]

    out_channels = [16, 32, 48, 64, 96, 128]
    target_outc = 3 * (5 + 1 + 10)
    fpn_in_list = out_channels[2:]
    fpn_out_list = out_channels[2:]

    classes = 1
    nanchors = 3
    face_anchors4 = [[[6,7],  [9,11],  [13,16]], [[18,23],  [26,33],  [37,47]], [[54,67],  [77,104],  [112,154]], [[174,238],  [258,355],  [445,568]]]
    # face_anchors3 = [[[4,5],  [8,10],  [13,16]], [[23,29],  [43,55],  [73,105]], [[146,217],  [231,300],  [335,433]]]
    # 1.25, 0.45, 2.8, 1.5, 3.75, 1.75, 5.9, 2.0, 6.8, 2.75 mark,ID
    # 0.31, 0.62, 0.45, 0.95, 0.55, 1.41, 0.62, 1.09, 0.78, 1.56 224*224 ears

    # data path
    data_list = "D:/data/imgs/widerface_clean/train_shuffle.txt"
    data_path = "D:/data/imgs/widerface_clean/train"

    model_save = "D:/codes/pytorch_projects/yolo_face/weights"
    pretrain = True
    pretrain_weights = "D:/codes/pytorch_projects/yolo_face/weights/FaceShuffle_62.pth"

    yaml_model = "D:/codes/pytorch_projects/yolo_face/models/yolov5-0.5.yaml"
    lr = 0.001
    weight_decay = 0.0005

    # test
    padding_img = True
    input_size = 1024  # net input 0: origin imgsize
    conf_thresh = 0.25
    nms_thresh = 0.3

facecfg = face_config()