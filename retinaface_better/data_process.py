import numpy as np
import os
import cv2
import random
import shutil

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

def create_FaceQuality_label(imgdirs, txtsave):
    label_classfication = open(txtsave, mode="w+")
    for root, dirs, files in os.walk(imgdirs):
        for file in files:
            rootsplit = root.replace('\\', '/').split("/")
            dir = rootsplit[-1]
            imgpath = dir + "/" + file
            splitfile = file.split(".")[0]
            namesplit = splitfile.split("_")
            savedata = imgpath + " " + namesplit[-1]
            label_classfication.write(savedata)
            label_classfication.write("\n")
    label_classfication.close()

def img_augment(imgdir, savedir):
    imgid = 750
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            root = root.replace('\\', '/')
            imgname, houzui = file.split(".")
            imgpath = root + "/" + file
            img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
            imgmirror = cv2.flip(img, 1)
            savepath1 = savedir + "/shadow" + str(imgid) + "_1." + houzui
            savepath2 = savedir + "/mshadow" + str(imgid) + "_1." + houzui
            cv2.imwrite(savepath1, img)
            cv2.imwrite(savepath2, imgmirror)
            imgid += 1

def img_fpn(imgdir, savedir):
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            root = root.replace('\\', '/')
            filename = file.split(".")[0]
            houzui = file.split(".")[1]
            imgpath = imgdir + "/" + file
            savep = savedir + "/" + filename + "_rs." + houzui

            img = cv2.imread(imgpath)
            imgh, imgw, imgc = img.shape
            maxwh = max(imgh, imgw)
            scale_size = random.choice([256, 356, 486, 555, 646, 880, 1011])
            scal = scale_size / maxwh
            imgsize = cv2.resize(img, None, None, fx=scal, fy=scal, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(savep, imgsize)
            # imgpath = root + "/" + file
            # img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
            # imgh, imgw, imgc = img.shape
            # maxwh = max(imgh, imgw)
            # if maxwh <= 500:
            #     add_facesize = random.choice([20, 30, 40])
            # if maxwh > 500 and maxwh <= 1500:
            #     add_facesize = random.choice([20, 30, 40, 50, 60, 80])
            # if maxwh > 1500 and maxwh <= 3000:
            #     add_facesize = random.choice([20, 30, 40, 50, 60, 80, 100, 120])
            # if maxwh > 3000:
            #     add_facesize = random.choice([20, 30, 40, 50, 60, 80, 100, 120, 150, 200])
            # scal = 20 / add_facesize
            # imgadd = cv2.resize(img, None, None, fx=scal, fy=scal, interpolation=cv2.INTER_CUBIC)
            #
            # savepath = savedir + "/" + file
            # cv2.imwrite(savepath, imgadd)

def get_img_withdir(imgdir1, imgdir2, save, dirnum=False):
    for root, dirs, files in os.walk(imgdir1):
        for file in files:
            if dirnum:
                root = root.replace("\\", "/")
                rootsplit = root.split("/")
                zidir = rootsplit[-1]
                imgpath = imgdir2 + "/" + zidir + "/" + file
                imgsave = save + "/" + file
                shutil.copy(imgpath, imgsave)
            else:
                imgpath = imgdir2 + "/" + file
                imgsave = save + "/" + file
                shutil.move(imgpath, imgsave)


def plot_label():
    txtp = "D:/data/imgs/facePicture/facepic/20210129/faces/28.txt"
    label_ = open(txtp, mode="r+")
    img_mat = cv2.imread("D:/data/imgs/facePicture/facepic/20210129/faces/28.jpg")

    lines = label_.readlines()
    for line in lines:
        line = line.rstrip()
        line = line.split(' ')
        label = [int(x) for x in line]
        fx = label[0]
        fy = label[1]
        fw = label[2]
        fh = label[3]

        cv2.rectangle(img_mat, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
        cv2.circle(img_mat, (label[4], label[5]), 1, (0, 0, 255), 4)
        cv2.circle(img_mat, (label[7], label[8]), 1, (0, 255, 255), 4)
        cv2.circle(img_mat, (label[10], label[11]), 1, (255, 0, 255), 4)
        cv2.circle(img_mat, (label[13], label[14]), 1, (0, 255, 0), 4)
        cv2.circle(img_mat, (label[16], label[17]), 1, (255, 0, 0), 4)
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.imshow('result', img_mat)
    cv2.waitKey(0)

def add_picTopic(imgp, addp, addtxt):
    imgadd = cv2.imread(addp)
    imgaddh, imgaddw, imgaddc = imgadd.shape
    maxwhadd = max(imgaddh, imgaddw)
    add_facesize = random.choice([40, 60, 80, 100, 150, 200, 250])
    add_scal = add_facesize / maxwhadd
    imgadd = cv2.resize(imgadd, None, None, fx=add_scal, fy=add_scal, interpolation=cv2.INTER_CUBIC)
    imgaddh, imgaddw, imgaddc = imgadd.shape

    img = cv2.imread(imgp)
    imgh, imgw, imgc = img.shape
    maxwh = max(imgh, imgw)
    _scal = 1024 / maxwh
    img = cv2.resize(img, None, None, fx=_scal, fy=_scal, interpolation=cv2.INTER_CUBIC)
    imgh, imgw, imgc = img.shape
    imgafteradd = img

    label_ = open(addtxt, mode="r+")
    lines = label_.readlines()
    savelabel = np.zeros(14, dtype=np.float32)
    for line in lines:
        addx = np.random.randint(0, imgw - imgaddw)
        addy = np.random.randint(0, imgh - imgaddh)
        line = line.rstrip()
        line = line.split(' ')
        label = [int(float(x) * add_scal) for x in line]
        fx = label[0] + addx
        fy = label[1] + addy
        fw = label[2]
        fh = label[3]
        lex = label[4] + addx
        ley = label[5] + addy
        rex = label[7] + addx
        rey = label[8] + addy
        nex = label[10] + addx
        ney = label[11] + addy
        lmx = label[13] + addx
        lmy = label[14] + addy
        rmx = label[16] + addx
        rmy = label[17] + addy
        savelabel[0] = fx
        savelabel[1] = fy
        savelabel[2] = fw
        savelabel[3] = fh
        savelabel[4] = lex
        savelabel[5] = ley
        savelabel[6] = rex
        savelabel[7] = rey
        savelabel[8] = nex
        savelabel[9] = ney
        savelabel[10] = lmx
        savelabel[11] = lmy
        savelabel[12] = rmx
        savelabel[13] = rmy
        imgafteradd[addy:addy + imgaddh, addx:addx + imgaddw, :] = imgadd
        # cv2.rectangle(imgafteradd, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
        # cv2.circle(imgafteradd, (lex, ley), 1, (0, 0, 255), 4)
        # cv2.circle(imgafteradd, (rex, rey), 1, (0, 255, 255), 4)
        # cv2.circle(imgafteradd, (nex, ney), 1, (255, 0, 255), 4)
        # cv2.circle(imgafteradd, (lmx, lmy), 1, (0, 255, 0), 4)
        # cv2.circle(imgafteradd, (rmx, rmy), 1, (255, 0, 0), 4)
    # cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    # cv2.imshow('result', imgafteradd)
    # cv2.waitKey(0)
    return imgafteradd, savelabel

def add_picTopicFPN(imgp, addp, addtxt):
    img = cv2.imread(imgp)
    imgh, imgw, imgc = img.shape
    maxwh = max(imgh, imgw)
    imgafteradd = img

    # if maxwh <= 320:
    #     add_facesize = random.choice([40, 60, 80, 90, 100])
    # if maxwh > 320 and maxwh <= 500:
    #     add_facesize = random.choice([40, 60, 80, 90, 100, 110, 120])
    # if maxwh > 500 and maxwh <= 850:
    #     add_facesize = random.choice([60, 80, 100, 120, 140, 150, 160, 180])
    # if maxwh > 850:
    #     add_facesize = random.choice([80, 100, 120, 150, 180, 200, 220, 230, 250, 260, 280, 300, 320, 340])

    add_facesize = random.choice([120, 140, 160, 190])
    # if maxwh == 256:
    #     add_facesize = 50
    # if maxwh == 356:
    #     add_facesize = 75
    # if maxwh == 486:
    #     add_facesize = 80
    # if maxwh == 646:
    #     add_facesize = 95
    # if maxwh == 880:
    #     add_facesize = 110
    # if maxwh == 1011:
    #     add_facesize = 150

    imgadd = cv2.imread(addp)
    imgaddh, imgaddw, imgaddc = imgadd.shape
    maxwhadd = max(imgaddh, imgaddw)
    add_scal = add_facesize / maxwhadd
    imgadd = cv2.resize(imgadd, None, None, fx=add_scal, fy=add_scal, interpolation=cv2.INTER_CUBIC)
    imgaddh, imgaddw, imgaddc = imgadd.shape

    label_ = open(addtxt, mode="r+")
    lines = label_.readlines()
    savelabel = np.zeros(14, dtype=np.float32)
    for line in lines:
        addx = np.random.randint(95, 220)
        addy = np.random.randint(95, 200)
        line = line.rstrip()
        line = line.split(' ')
        label = [int(float(x) * add_scal) for x in line]
        fx = label[0] + addx
        fy = label[1] + addy
        fw = label[2]
        fh = label[3]
        lex = label[4] + addx
        ley = label[5] + addy
        rex = label[7] + addx
        rey = label[8] + addy
        nex = label[10] + addx
        ney = label[11] + addy
        lmx = label[13] + addx
        lmy = label[14] + addy
        rmx = label[16] + addx
        rmy = label[17] + addy
        savelabel[0] = fx
        savelabel[1] = fy
        savelabel[2] = fw
        savelabel[3] = fh
        savelabel[4] = lex
        savelabel[5] = ley
        savelabel[6] = rex
        savelabel[7] = rey
        savelabel[8] = nex
        savelabel[9] = ney
        savelabel[10] = lmx
        savelabel[11] = lmy
        savelabel[12] = rmx
        savelabel[13] = rmy
        imgafteradd[addy:addy + imgaddh, addx:addx + imgaddw, :] = imgadd
    return imgafteradd, savelabel

def create_addpics(imgdir, facedir, savedir, labtxt):
    label_face = open(labtxt, mode="w+")
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            path_head = "# addbg/" + file
            label_face.write(path_head + "\n")
            faceid = random.randint(0, 14)
            facepath = facedir + "/" + str(faceid) + ".jpg"
            facetxt = facedir + "/" + str(faceid) + ".txt"

            imgpath = imgdir + "/" + file
            # img, lab = add_picTopic(imgpath, facepath, facetxt)
            img, lab = add_picTopicFPN(imgpath, facepath, facetxt)

            fx = ('%d' % lab[0])
            fy = ('%d' % lab[1])
            fw = ('%d' % lab[2])
            fh = ('%d' % lab[3])
            lex = ('%d' % lab[4])
            ley = ('%d' % lab[5])
            rex = ('%d' % lab[6])
            rey = ('%d' % lab[7])
            nex = ('%d' % lab[8])
            ney = ('%d' % lab[9])
            lmx = ('%d' % lab[10])
            lmy = ('%d' % lab[11])
            rmx = ('%d' % lab[12])
            rmy = ('%d' % lab[13])
            face_pos = fx + " " + fy + " " + fw + " " + fh + " " + lex + " " + ley + " 0 " + rex + " " + rey + " 0 " + nex + " " + ney + " 0 " + lmx + " " + lmy + " 0 " + rmx + " " + rmy + " 0 1"
            label_face.write(face_pos + "\n")

            imgsave = savedir + "/" + file
            cv2.imwrite(imgsave, img)
    label_face.close()

def resize_imgfpn_dir(imgdir, txtdir, savedir, labtxt):
    label_face = open(labtxt, mode="w+")
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            filename = file.split(".")[0]
            path_head = "# addface5/" + file
            label_face.write(path_head + "\n")

            imgpath = imgdir + "/" + file
            savepath = savedir + "/" + file
            txtpath = txtdir + "/" + filename + ".txt"

            img = cv2.imread(imgpath)
            imgh, imgw, imgc = img.shape
            maxwh = max(imgh, imgw)
            imgsize = random.choice([300, 360, 400, 450, 480, 520, 560, 600, 650, 720, 800, 1000])
            scal = imgsize / maxwh
            imgres = cv2.resize(img, None, None, fx=scal, fy=scal, interpolation=cv2.INTER_CUBIC)

            label_ = open(txtpath, mode="r+")
            lines = label_.readlines()
            for line in lines:
                if line.startswith('#'):
                    continue
                line = line.rstrip()
                line = line.split(' ')
                label = [int(float(x) * scal) for x in line]
                fx = ('%d' % label[0])
                fy = ('%d' % label[1])
                fw = ('%d' % label[2])
                fh = ('%d' % label[3])
                lex = ('%d' % label[4])
                ley = ('%d' % label[5])
                rex = ('%d' % label[7])
                rey = ('%d' % label[8])
                nex = ('%d' % label[10])
                ney = ('%d' % label[11])
                lmx = ('%d' % label[13])
                lmy = ('%d' % label[14])
                rmx = ('%d' % label[16])
                rmy = ('%d' % label[17])
                face_pos = fx + " " + fy + " " + fw + " " + fh + " " + lex + " " + ley + " 0 " + rex + " " + rey + " 0 " + nex + " " + ney + " 0 " + lmx + " " + lmy + " 0 " + rmx + " " + rmy + " 0 1"
                label_face.write(face_pos + "\n")

            cv2.imwrite(savepath, imgres)
            label_.close()
    label_face.close()

def save_label_to_txt(path_head, txtp, scale, labfile):
    labfile.write(path_head + "\n")

    label_ = open(txtp, mode="r+")
    lines = label_.readlines()
    for line in lines:
        if line.startswith('#'):
            continue
        line = line.rstrip()
        line = line.split(' ')
        label = [int(float(x) * scale) for x in line]
        fx = ('%d' % label[0])
        fy = ('%d' % label[1])
        fw = ('%d' % label[2])
        fh = ('%d' % label[3])
        lex = ('%d' % label[4])
        ley = ('%d' % label[5])
        rex = ('%d' % label[7])
        rey = ('%d' % label[8])
        nex = ('%d' % label[10])
        ney = ('%d' % label[11])
        lmx = ('%d' % label[13])
        lmy = ('%d' % label[14])
        rmx = ('%d' % label[16])
        rmy = ('%d' % label[17])
        face_pos = fx + " " + fy + " " + fw + " " + fh + " " + lex + " " + ley + " 0 " + rex + " " + rey + " 0 " + nex + " " + ney + " 0 " + lmx + " " + lmy + " 0 " + rmx + " " + rmy + " 0 1"
        labfile.write(face_pos + "\n")
    label_.close()

def resize_img_and_label(imgdir, txtdir, savedir, labtxt):
    label_face = open(labtxt, mode="w+")
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            filename = file.split(".")[0]
            houzui = file.split(".")[1]

            head256 = "# addface7/" + filename + "_256." + houzui
            head356 = "# addface7/" + filename + "_356." + houzui
            head486 = "# addface7/" + filename + "_486." + houzui
            head646 = "# addface7/" + filename + "_646." + houzui
            head880 = "# addface7/" + filename + "_880." + houzui

            imgpath = imgdir + "/" + file
            save256 = savedir + "/" + filename + "_256." + houzui
            save356 = savedir + "/" + filename + "_356." + houzui
            save486 = savedir + "/" + filename + "_486." + houzui
            save646 = savedir + "/" + filename + "_646." + houzui
            save880 = savedir + "/" + filename + "_880." + houzui
            txtpath = txtdir + "/" + filename + ".txt"

            img = cv2.imread(imgpath)
            imgh, imgw, imgc = img.shape
            maxwh = max(imgh, imgw)
            scal256 = 256 / maxwh
            scal356 = 356 / maxwh
            scal486 = 486 / maxwh
            scal646 = 646 / maxwh
            scal880 = 880 / maxwh
            img256 = cv2.resize(img, None, None, fx=scal256, fy=scal256, interpolation=cv2.INTER_CUBIC)
            img356 = cv2.resize(img, None, None, fx=scal356, fy=scal356, interpolation=cv2.INTER_CUBIC)
            img486 = cv2.resize(img, None, None, fx=scal486, fy=scal486, interpolation=cv2.INTER_CUBIC)
            img646 = cv2.resize(img, None, None, fx=scal646, fy=scal646, interpolation=cv2.INTER_CUBIC)
            img880 = cv2.resize(img, None, None, fx=scal880, fy=scal880, interpolation=cv2.INTER_CUBIC)

            save_label_to_txt(head256, txtpath, scal256, label_face)
            cv2.imwrite(save256, img256)
            save_label_to_txt(head356, txtpath, scal356, label_face)
            cv2.imwrite(save356, img356)
            save_label_to_txt(head486, txtpath, scal486, label_face)
            cv2.imwrite(save486, img486)
            save_label_to_txt(head646, txtpath, scal646, label_face)
            cv2.imwrite(save646, img646)
            save_label_to_txt(head880, txtpath, scal880, label_face)
            cv2.imwrite(save880, img880)

    label_face.close()

def sum_add_label(imgdir, txtdir, labtxt):
    label_face = open(labtxt, mode="w+")
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            filename = file.split(".")[0]
            houzui = file.split(".")[1]
            path_head = "# face320/" + file
            label_face.write(path_head + "\n")

            txtpath = txtdir + "/" + filename + ".txt"
            label_ = open(txtpath, mode="r+")
            lines = label_.readlines()
            for line in lines:
                if line.startswith('#'):
                    continue
                line = line.rstrip()
                line = line.split(' ')
                label = [round(float(x)) for x in line]
                fx = ('%d' % label[0])
                fy = ('%d' % label[1])
                fw = ('%d' % label[2])
                fh = ('%d' % label[3])
                lex = ('%d' % label[4])
                ley = ('%d' % label[5])
                rex = ('%d' % label[7])
                rey = ('%d' % label[8])
                nex = ('%d' % label[10])
                ney = ('%d' % label[11])
                lmx = ('%d' % label[13])
                lmy = ('%d' % label[14])
                rmx = ('%d' % label[16])
                rmy = ('%d' % label[17])
                face_bg = ('%d' % label[19])
                face_pos = fx + " " + fy + " " + fw + " " + fh + " " + lex + " " + ley + " 0 " + rex + " " \
                           + rey + " 0 " + nex + " " + ney + " 0 " + lmx + " " + lmy + " 0 " + rmx + " " + rmy + " 0 " + face_bg
                label_face.write(face_pos + "\n")
            label_.close()
    label_face.close()

def imgfpn_label():
    imgadd = cv2.imread("D:/data/imgs/facePicture/facepic/20210129/addface2/add2.png")
    label_ = open("D:/data/imgs/facePicture/facepic/20210129/addlabel2.txt", mode="r+")
    lines = label_.readlines()
    idd = 1

    label_face = open("D:/data/imgs/facePicture/facepic/20210129/addlabel22.txt", mode="w+")

    for i in range(150):
        add_facesize = random.choice(
            [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200])
        add_scal = 20 / add_facesize
        imgs = cv2.resize(imgadd, None, None, fx=add_scal, fy=add_scal, interpolation=cv2.INTER_CUBIC)

        imgh, imgw, imgc = imgs.shape

        if imgw > 280 or imgh > 280:
            path_head = "# addface/" + "add2" + str(idd) + ".png"
            label_face.write(path_head + "\n")

            for line in lines:
                if line.startswith('#'):
                    continue
                line = line.rstrip()
                line = line.split(' ')
                label = [str(int(float(x) * add_scal)) for x in line]

                face_pos = label[0] + " " + label[1] + " " + label[2] + " " + label[3] + " " + label[4] + " " + label[
                    5] + " 0 " + \
                           label[7] + " " + label[8] + " 0 " + label[10] + " " + label[11] + " 0 " + label[13] + " " + \
                           label[14] + " 0 " + label[16] + " " + label[17] + " 0 1"
                label_face.write(face_pos + "\n")

            savepath = "D:/data/imgs/facePicture/facepic/20210129/addface2" + "/" + "add2" + str(idd) + ".png"
            cv2.imwrite(savepath, imgs)
            idd += 1
    label_face.close()

def get_patches_img(imgpath, savedir, patch_num=15):
    img = cv2.imread(imgpath)
    for i in range(patch_num):
        x1 = np.random.randint(0, 180)
        y1 = np.random.randint(0, 250)
        x2 = np.random.randint(450, 1200)
        y2 = np.random.randint(510, 800)
        patch_img = img[y1:y2, x1:x2, :]
        savepath = savedir + "/" + str(i) + "wj.jpg"
        cv2.imwrite(savepath, patch_img)

def delete_img(errdir, imgdir):
    for root, dirs, files in os.walk(errdir):
        for file in files:
            imgpath = imgdir + "/" + file
            os.remove(imgpath)

def add_background(imgdir, labtxt):
    label_face = open(labtxt, mode="w+")
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            path_head = "# noface/" + file
            label_face.write(path_head + "\n")

            face_pos = "0 0 0 0 -1 -1 0 -1 -1 0 -1 -1 0 -1 -1 0 -1 -1 0 1"
            label_face.write(face_pos + "\n")
    label_face.close()

def widerface_to_yolo(imgdir, txtdir, yolotxt, yolotxtpath):
    yolo_txt_path = open(yolotxtpath, mode="w+")
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            filename = file.split(".")[0]
            houzui = file.split(".")[1]
            root = root.replace("\\", "/")
            imgpath = root + "/" + file
            img = cv2.imread(imgpath)
            imgh, imgw, imgc = img.shape

            txtpath = txtdir + "/" + filename + ".txt"
            yolopath = yolotxt + "/" + filename + ".txt"
            yolo_txt_path.write(yolopath + "\n")
            label_txt = open(txtpath, mode="r+")
            yolo_txt = open(yolopath, mode="w+")
            lines = label_txt.readlines()
            for line in lines:
                if line.startswith('#'):
                    continue
                line = line.rstrip()
                line = line.split(' ')
                label = [float(x)for x in line]
                cx = label[0] + 0.5 * label[2]
                cy = label[1] + 0.5 * label[3]
                iw = label[2]
                ih = label[3]
                cx = cx / imgw
                cy = cy / imgh
                iw = iw / imgw
                ih = ih / imgh
                fx = ('%.4f' % cx)
                fy = ('%.4f' % cy)
                fw = ('%.4f' % iw)
                fh = ('%.4f' % ih)
                face_pos = "1" + " " + fx + " " + fy + " " + fw + " " + fh
                yolo_txt.write(face_pos + "\n")
            label_txt.close()
            yolo_txt.close()
    yolo_txt_path.close()

def move_yolotxt_to_imgdir(imgdir, yolotxtd, yolop):
    yolo_txt_path = open(yolop, mode="w+")
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            filename = file.split(".")[0]
            houzui = file.split(".")[1]
            root = root.replace("\\", "/")
            txtpath = yolotxtd + "/" + filename + ".txt"
            movepath = root + "/" + filename + ".txt"
            shutil.move(txtpath, movepath)
            yolo_txt_path.write(movepath + "\n")
    yolo_txt_path.close()

def move_selectimg_fromdir(selectdir, imgdir, savedir):
    for root, dirs, files in os.walk(selectdir):
        for file in files:
            file_split = file.split("_")
            # subdir = file_split[0]
            # imgname = file_split[1]
            # srcp = imgdir + "/" + subdir + "/" + imgname
            srcpp = imgdir + "/" + file
            dstp = savedir + "/" + file
            shutil.move(srcpp, dstp)

def move_wantedimg_fromdir(imgdir, savedir):
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            srcpp = imgdir + "/" + file
            img = cv2.imread(srcpp)
            cv2.imshow('result', img)
            if cv2.waitKey(0) == 121:
                dstp = savedir + "/" + file
                shutil.move(srcpp, dstp)
            else:
                continue

def remove_space(imgdirs):
    for root, dirs, files in os.walk(imgdirs):
        for file in files:
            root = root.replace('\\', '/')
            file_nospace = file.replace(" ", "")
            imgpath = root + "/" + file
            replace_path = root + "/" + file_nospace
            # imgpath = imgdirs + "/" + file
            # replace_path = imgdirs + "/" + file_nospace
            os.rename(imgpath, replace_path)

def class_images(imgdir, classd1):
    cv2.namedWindow('show', cv2.WINDOW_NORMAL)
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            imgpath = imgdir + "/" + file
            img_mat = cv2.imread(imgpath)
            cv2.imshow('show', img_mat)
            if cv2.waitKey(0) == 121:
                continue
            else:
                dstp = classd1 + "/" + file
                shutil.move(imgpath, dstp)

def get_img_list(imgdir, listpath, subdir=False):
    list_file = open(listpath, 'w+')
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            root = root.replace('\\', '/')
            if subdir:
                imgpath = root + "/" + file
                list_file.write(imgpath + "\n")
            else:
                imgpath = imgdir + "/" + file
                list_file.write(imgpath + "\n")
    list_file.close()

def get_txt_byimg(imgdir, txtdir, savedir):
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            fiename = file.split(".")[0]
            txtpath = txtdir + "/" + fiename + ".txt"
            txtsave = savedir + "/" + fiename + ".txt"
            shutil.move(txtpath, txtsave)

def tonjiFaceWH(txt_path, listpath):
    list_file = open(listpath, 'w+')
    witedata = "facew   faceh" + "\n"
    list_file.write(witedata)
    num8 = 0
    num820 = 0
    num2040 = 0
    num4080 = 0
    num80120 = 0
    num120180 = 0
    num180250 = 0
    num250300 = 0
    num300350 = 0
    num350400 = 0
    num400450 = 0
    num450500 = 0
    num500550 = 0
    num550600 = 0
    num600650 = 0
    num650700 =0
    num700 = 0

    f = open(txt_path, 'r')
    lines = f.readlines()
    isFirst = True
    for line in lines:
        line = line.rstrip()
        if line.startswith('#'):
            if isFirst is True:
                isFirst = False

            name = line[2:]
            # imgpath = txt_path.replace('train_resize.txt', 'images/') + name
            # img = cv2.imread(imgpath)
            # ih, iw, _ = img.shape
        else:
            line = line.split(' ')
            label = [float(x) for x in line]
            w = str(label[2])
            h = str(label[3])
            if label[2] > 1300 or label[3] > 1300: #w < 5 or h < 5 or w > 1000 or h > 1000:
                print(name)
                continue
            # witewh = w + "   " + h + "   " + str(iw) + "   " + str(ih) + "\n"
            witewh = w + "   " + h + "   " + "\n"
            list_file.write(witewh)
            # maxwh = max(w, h)
            # if maxwh < 8:
            #     num8 += 1
            # if maxwh >= 8 and maxwh < 20:
            #     num820 += 1
            # if maxwh >= 20 and maxwh < 40:
            #     num2040 += 1
            # if maxwh >= 40 and maxwh < 80:
            #     num4080 += 1
            # if maxwh >= 80 and maxwh < 120:
            #     num80120 += 1
            # if maxwh >= 120 and maxwh < 180:
            #     num120180 += 1
            # if maxwh >= 180 and maxwh < 250:
            #     num180250 += 1
            # if maxwh >= 250 and maxwh < 300:
            #     num250300 += 1
            # if maxwh >= 300 and maxwh < 350:
            #     num300350 += 1
            # if maxwh >= 350 and maxwh < 400:
            #     num350400 += 1
            # if maxwh >= 400 and maxwh < 450:
            #     num400450 += 1
            # if maxwh >= 450 and maxwh < 500:
            #     num450500 += 1
            # if maxwh >= 500 and maxwh < 550:
            #     num500550 += 1
            # if maxwh >= 550 and maxwh < 600:
            #     num550600 += 1
            # if maxwh >= 600 and maxwh < 650:
            #     num600650 += 1
            # if maxwh >= 650 and maxwh < 700:
            #     num650700 += 1
            # if maxwh >= 700:
            #     num700 += 1
    list_file.close()


if __name__ == "__main__":
    imgpath = "D:/data/imgs/facePicture/blur/faces"
    txtpath = "D:/data/imgs/widerface_clean/train_resize.txt"
    # create_FaceQuality_label(imgpath, txtpath)

    txtpath2 = "D:/data/imgs/widerface_clean/faceWH_tain2.txt"
    shufflepath = "D:/data/imgs/facePicture/blur/blur1_shuffle.txt"
    # shuffle_txt(txtpath2, shufflepath)
    tonjiFaceWH(txtpath, txtpath2)

    dirp = "D:/data/imgs/ornament_IDphoto/train/0"
    dirs = "D:/data/imgs/widerface_clean/txt"
    savedir = "D:/data/imgs/widerface_clean/txt2"
    # get_img_list(dirp, txtpath, subdir=False)
    # get_txt_byimg(dirp, dirs, savedir)
    # class_images(dirp, savedir)
    # img_augment(dirp, dirs)
    # img_fpn(dirp, dirs)
    # get_img_withdir(dirp, dirs, savedir)
    # get_img_withdir(dirp, dirs, savedir, dirnum=False)
    # remove_space(dirp)
    # plot_label()

    # imgp1 = "D:/data/imgs/facePicture/facepic/20210129/noface/0a16a25467ee4eab80874a1fea24605d.png"
    # imgadd1 = "D:/data/imgs/facePicture/facepic/20210129/faces/4.jpg"
    # txtadd = "D:/data/imgs/facePicture/facepic/20210129/faces/4.txt"
    # img, lab = add_picTopic(imgp1, imgadd1, txtadd)
    # cv2.rectangle(img, (lab[0], lab[1]), (lab[0] + lab[2], lab[1] + lab[3]), (0, 255, 0), 2)
    # cv2.circle(img, (lab[4], lab[5]), 1, (0, 0, 255), 4)
    # cv2.circle(img, (lab[6], lab[7]), 1, (0, 255, 255), 4)
    # cv2.circle(img, (lab[8], lab[9]), 1, (255, 0, 255), 4)
    # cv2.circle(img, (lab[10], lab[11]), 1, (0, 255, 0), 4)
    # cv2.circle(img, (lab[12], lab[13]), 1, (255, 0, 0), 4)
    # cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    # cv2.imshow('result', img)
    # cv2.waitKey(0)

    imgd = "D:/data/imgs/facePicture/facepic/20210517/face320_wanted" # D:/data/imgs/facePicture/facepic/20210129
    txtd = "D:/data/imgs/facePicture/facepic/aatxt/face320_wanted"
    faced = "D:/data/imgs/facePicture/facepic/20210129/faces2"
    saved = "D:/data/imgs/facePicture/facepic/20210129/addface9/addface10"
    labp = "D:/data/imgs/facePicture/facepic/aatxt/face320_better.txt"
    # create_addpics(imgd, faced, saved, labp)
    # add_background(imgd, labp)
    # resize_imgfpn_dir(imgd, txtd, saved, labp)
    imgpp = "D:/data/imgs/facePicture/facepic/20210129/addface6/cc.png"
    # get_patches_img(imgpp, saved, patch_num=300)
    # resize_img_and_label(imgd, txtd, saved, labp)
    # sum_add_label(imgd, txtd, labp)

    deled = "D:/data/imgs/facePicture/facepic/20210129/addface11/add_new"  # D:/data/imgs/facePicture/facepic/20210129
    errord = "D:/data/imgs/facePicture/facepic/20210129/addface11/add_wujian/fb"
    # delete_img(errord, deled)

    widerd = "D:/data/imgs/facePicture/facepic/20210129/addface11"
    widertxt = "D:/data/imgs/facePicture/facepic/20210129/addface12/alltxt"
    yolod = "D:/data/imgs/facePicture/facepic/20210129/addface12/yolotxt"
    yolotxtp = "D:/data/imgs/facePicture/facepic/20210129/addface12/wideryolo2.txt"
    selectd = "D:/data/imgs/facePicture/facepic/20210129/addface13/error"
    origind = "D:/data/imgs/ornament_IDphoto/part1"
    save_selectd = "D:/data/imgs/ornament_IDphoto/select"
    # widerface_to_yolo(widerd, widertxt, yolod, yolotxtp)
    # move_yolotxt_to_imgdir(widerd, yolod, yolotxtp)
    # move_selectimg_fromdir(selectd, origind, save_selectd)
    # move_wantedimg_fromdir(origind, save_selectd)

















