import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

classes = ["face"]  # 人脸检测


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id):
    in_file = open('F:/tanbing/vocxml/%s.xml' % (image_id))

    out_file = open('F:/tanbing/cardata/%s.txt' % (image_id), 'w')  # 生成txt格式文件
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

def xml_to_widerface(xmllist, txtp):
    xmls = open(xmllist, 'r+')
    labtxt = open(txtp, mode='w+')
    for line in xmls:
        line = line.strip('\n')
        linesplit = line.split("/")
        dirname = linesplit[-3]
        imgname = linesplit[-1].split(".")[0] + ".jpg"
        head = "# " + dirname + "/" + imgname + "\n"
        labtxt.write(head)

        tree = ET.parse(line)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))


# image_ids_val = open('/home/*****/darknet/scripts/VOCdevkit/voc/list').read().strip().split()

if __name__ == "__main__":
    image_ids_train = open('F:/tanbing/yolotxt/carxml.list').read().strip().split()  # list格式只有000000 000001
    for image_id in image_ids_train:
        convert_annotation(image_id)