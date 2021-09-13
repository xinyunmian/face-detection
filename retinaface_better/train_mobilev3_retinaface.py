from __future__ import print_function
import os
import torch
import cv2
import torch.optim as optim
import argparse
import torch.utils.data as data
from load_data import FaceDataLoader, Loader_collate
from data_augment import preproc
from retinaface_loss import MultiBoxLoss
from create_anchors import PriorBox
from config import cfg_slimNet3 as cfg_slimNet
from slim_net import FaceDetectSlimNet
import time
import datetime
import math
from collections import OrderedDict

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--training_dataset', default='D:/data/imgs/widerface_clean/train_resize.txt', help='Training dataset directory')
parser.add_argument('--test_dataset', default='D:/data/imgs/facePicture/facepic/20210129/addface9/aaa', help='Test dataset directory')
parser.add_argument('--test_minface', default=20, type=int, help='minface to detect')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default="D:/codes/pytorch_projects/retinaface_better/weights/", help='Location to save checkpoint models')

args = parser.parse_args()

torch.cuda.set_device(0)
#config some parameters
cfg = cfg_slimNet
num_classes = cfg['num_classes']
img_dim = cfg['image_size']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
training_dataset = args.training_dataset
save_folder = args.save_folder

# test
test_dir = args.test_dataset
min_face = args.test_minface

def load_pretrain(net):
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

def test_model(dnet, imgdir, minface):
    from test import test_one
    dnet.eval()
    allfaceNum = 0
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            imgpath = imgdir + "/" + file
            saveimg, facenum = test_one(imgpath, dnet, minface, dir=True)
            allfaceNum += facenum
    return allfaceNum

def adjust_learning_rate(epoch, optimizer):
    lr = args.lr
    if epoch > 650:
        lr = lr / 1000000
    elif epoch > 600:
        lr = lr / 100000
    elif epoch > 550:
        lr = lr / 10000
    elif epoch > 150:
        lr = lr / 1000
    elif epoch > 80:
        lr = lr / 100
    elif epoch > 20:
        lr = lr / 10
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr

def train():
    net = FaceDetectSlimNet(cfg=cfg_slimNet)
    if args.resume_net is not None:
        load_pretrain(net)

    net = net.cuda()
    print("to cuda done")
    criterion = MultiBoxLoss(num_classes, 0.4, 6)
    priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
    # optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)

    optimizer = torch.optim.SGD([
                    {'params': net.parameters()},
                    {'params': criterion.parameters(),}
                ], lr=initial_lr, momentum=momentum, weight_decay=weight_decay)

    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.cuda()
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')
    # cv2.namedWindow("show_img", cv2.WINDOW_KEEPRATIO)
    dataset = FaceDataLoader(training_dataset, preproc(img_dim))

    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=Loader_collate))
            if (epoch % 2 == 0 and epoch > 0):
                # torch.save(net.state_dict(), save_folder + "face_slim" + '_0526_' + str(epoch) + '.pth', _use_new_zipfile_serialization=False)
                torch.save(net.state_dict(), save_folder + "face_slim" + '_0609_' + str(epoch) + '.pth')
            # torch.save(net.state_dict(), save_folder + 'epoch_0526.pth', _use_new_zipfile_serialization=False)
            torch.save(net.state_dict(), save_folder + 'epoch_0609.pth')
            epoch += 1

        load_t0 = time.time()
        # 调用学习率调整函数
        lr_cur = adjust_learning_rate(epoch, optimizer)

        images, targets = next(batch_iterator)
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]
        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm, loss = criterion(out, priors, targets)
        # loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        print("Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Class: {:.4f} Loc: {:.4f} Landm: {:.4f} Sum: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s"
              .format(epoch, max_epoch, (iteration % epoch_size) + 1, epoch_size, iteration + 1, max_iter,
                      loss_c.item(), loss_l.item(), loss_landm.item(), loss.item(), lr_cur, batch_time))
    # torch.save(net.state_dict(), save_folder + 'final_0526.pth', _use_new_zipfile_serialization=False)
    torch.save(net.state_dict(), save_folder + 'final_0609.pth')


if __name__ == "__main__":
    train()



















































