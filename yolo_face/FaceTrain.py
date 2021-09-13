import torch
import os
import random
import torch.utils.data as data
from FaceDataLoader import LoadFace, collate_face
from FaceNet import FaceYoloNet
from FaceConfig import facecfg
from FaceLoss import compute_FaceLoss
#cuda
torch.cuda.set_device(0)

def adjust_learning_rate(epoch, optimizer):
    lr = facecfg.lr
    if epoch > 650:
        lr = lr / 1000000
    elif epoch > 600:
        lr = lr / 100000
    elif epoch > 550:
        lr = lr / 10000
    elif epoch > 350:
        lr = lr / 1000
    elif epoch > 200:
        lr = lr / 100
    elif epoch > 20:
        lr = lr / 10
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def train():
    data_train = LoadFace(img_path=facecfg.data_list, cfg=facecfg)
    train_loader = data.DataLoader(data_train, batch_size=facecfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate_face)

    net = FaceYoloNet(cfg=facecfg)
    net.cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=facecfg.lr, weight_decay=facecfg.weight_decay, momentum=0.9)
    net.train()

    for epoch in range(facecfg.start_epoch, facecfg.epochs):
        print("Epoch {}".format(epoch))
        print('I am training, please wait...')
        batch = 0
        lr_cur = 0.01
        for i, (imgs, targets, paths, _) in enumerate(train_loader):
            imgs = imgs.cuda().float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
            targets = targets.cuda()
            out8, out16, out32, out64 = net(imgs)
            outs = []
            outs.append(out8)
            outs.append(out16)
            outs.append(out32)
            outs.append(out64)
            loss, loss_items = compute_FaceLoss(outs, targets, facecfg.face_anchors4, wh_r=4, downs=[8, 16, 32, 64])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for params in optimizer.param_groups:
                lr_cur = params['lr']
            if batch % 5 == 0:
                print("Epoch:{}/{} || Loss:{:.4f} || LR: {:.8f}".format(epoch, facecfg.epochs, loss.item(), lr_cur))

        # 调用学习率调整函数
        adjust_learning_rate(epoch, optimizer)
        if (epoch % 2 == 0 and epoch > 0):
            torch.save(net.state_dict(), facecfg.model_save + "/" + "FaceDetect_{}.pth".format(epoch))


if __name__ == "__main__":
    train()











