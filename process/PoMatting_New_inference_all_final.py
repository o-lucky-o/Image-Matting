import sys
from PyQt5 import QtGui
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QLabel, QApplication

import os
import cv2
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from src.models.modnet import MODNet
from PyQt5.QtCore import pyqtSignal
from PIL.ImageQt import toqpixmap
from threading import Thread
import time

from PyQt5.QtGui import *

class MyWindow(QMainWindow):
    displayEmitApp = pyqtSignal(str)
    displayEmitApp2 = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):

        self.input_path = None

        # 窗口的大小
        self.resize(1650, 600)  # 宽800(x), 高500(y)
        # self.resize(750, 300) # 宽800(x), 高500(y)

        # 设置窗口标题
        self.setWindowTitle("PortraitMatting")

        # 调用父类中的menuBar，从而对菜单栏进行操作
        menu = self.menuBar()
        # 如果是Mac的话，菜单栏不会在Window中显示而是屏幕顶部系统菜单栏位置
        # 下面这一行代码使得Mac也按照Windows的那种方式在Window中显示Menu
        menu.setNativeMenuBar(False)

        image_menu = menu.addMenu("图片")
        i_open = image_menu.addAction("打开")
        i_save = image_menu.addAction("保存")

        video_menu = menu.addMenu("视频")
        v_open = video_menu.addAction("打开")
        v_save = video_menu.addAction("保存")

        # 纯文本
        path = QLabel("文件路径：", self)
        path.setGeometry(10, 50, 50, 20)

        # 纯文本
        self.path_display = QLabel(self)
        self.path_display.setText("")
        self.path_display.setGeometry(60, 50, 1540, 20)  # 横着的x，竖着的y，长度，宽度像素点
        # self.path_display.setGeometry(60, 50, 680, 20)  # 横着的x，竖着的y，长度，宽度像素点
        # self.path_display.move(100, 50)
        # 字体设置
        self.path_display.setStyleSheet("QLabel{background:white;}"
                                        "QLabel{color:rgb(300,300,300,120);font-size:15px;font-weight:bold;font-family:宋体;}")

        # 纯文本
        self.input = QLabel(self)
        self.input.setText("")
        self.input.setGeometry(60, 80, 750, 500)  # * , * 长度(右)，宽度(左)
        # self.input.setGeometry(60, 80, 300, 200)  # * , * 长度(右)，宽度(左)
        # self.label.setFixedSize(int(410 * 2 / 3), int(410 * 2 / 3))
        # self.input.move(160, 100)
        self.input.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:15px;font-weight:bold;font-family:宋体;}")

        # 纯文本
        self.output = QLabel(self)
        self.output.setText("")
        self.output.setGeometry(850, 80, 750, 500)  # * , * 长度(右)，宽度(左)
        # self.output.setGeometry(440, 80, 300, 200)  # * , * 长度(右)，宽度(左)
        # self.label.setFixedSize(int(410 * 2 / 3), int(410 * 2 / 3))
        # self.output.move(160, 100)
        self.output.setStyleSheet("QLabel{background:white;}"
                                  "QLabel{color:rgb(300,300,300,120);font-size:15px;font-weight:bold;font-family:宋体;}")

        #
        # self.path_display = QLineEdit(self)
        i_open.triggered.connect(self.openimage)
        self.displayEmitApp.connect(self.inference)

        #v_open.triggered.connect(self.inputvideo)
        v_open.triggered.connect(self.displayPath)

        #self.displayEmitApp2.connect(self.outputvideo)
        self.displayEmitApp2.connect(self.inputvideo)
        self.displayEmitApp2.connect(self.startthread)

    def startthread(self):
        th = Thread(target= self.outputvideo)  # https://www.bbsmax.com/A/kmzLkAjKdG/
        th.start()
        time.sleep(0.01)


    def displayPath(self):
        self.vIn, imgType = QFileDialog.getOpenFileName(self, "打开mp4", "", "*.mp4;;*.png;;All Files(*)")
        self.path_display.setText(self.vIn)
        self.displayEmitApp2.emit()
        # th = Thread(target=self.outputvideo)  # https://www.bbsmax.com/A/kmzLkAjKdG/
        # th.start()
        # time.sleep(0.01)

    def inputvideo(self):
        self.vInPath = self.vIn
        self.cap_in = cv2.VideoCapture(self.vInPath)
        self.frameRate = 20
        # # 视频流设置 参数详见https://github.com/opencv/opencv/blob/master/modules/videoio/include/opencv2/videoio.hpp
        self.cap_in.set(cv2.CAP_PROP_FPS, self.frameRate) # 帧率 帧 / 秒   https: // www.javaroad.cn / questions / 288491

        while self.cap_in.isOpened():
            success, frame = self.cap_in.read()
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #frame = cv2.resize(frame,(self.input.height(),self.input.width(),3))

            if success == False:
                continue
            img = QImage(frame.data, self.input.width(), self.input.height(), QImage.Format_RGB888)
            #img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            #mywindow.SetPic(a)
            self.input.setPixmap(QPixmap.fromImage(img))

            cv2.waitKey(int(1000 / self.frameRate))

    def outputvideo(self):

        # Path = self.vInPath.spilt("/")[-2]
        # imgName = self.vInPath.spilt("/")[-1].spilt('.')[0]
        # outPath = os.path.join()
        # self.vInPath
        # self.cap_out = cv2.VideoCapture()  # 或读取相机cap = cv2.VideoCapture(0)
        Path = self.vIn.strip().rsplit("/", 2)
        outName = Path[2].split(".")[0] + '_fg' + '.mp4'
        self.vOutPath = os.path.join(Path[0], 'output_video', outName)

        self.cap_out = cv2.VideoCapture(self.vOutPath )
        self.frameRate = 20
        # # 视频流设置 参数详见https://github.com/opencv/opencv/blob/master/modules/videoio/include/opencv2/videoio.hpp
        self.cap_out.set(cv2.CAP_PROP_FPS, self.frameRate)  # 帧率 帧 / 秒   https: // www.javaroad.cn / questions / 288491

        while self.cap_out.isOpened():
            success, frame = self.cap_out.read()
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame = cv2.resize(frame,(self.input.height(),self.input.width()))

            if success == False:
                continue
            # img = QImage(frame.data, self.input.width(), self.input.height(), QImage.Format_RGB888)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            # mywindow.SetPic(a)
            self.output.setPixmap(QPixmap.fromImage(img))

            cv2.waitKey(int(1000 / self.frameRate))
        





    def openimage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")

        jpg = QtGui.QPixmap(imgName).scaled(self.input.width(), self.input.height())
        self.input.setPixmap(jpg)
        self.path_display.setText(imgName)
        self.displayEmitApp.emit(imgName)
        # out = self.inference(imgName)
        # foreground = out.toqpixmap()
        # self.output.setPixmap(foreground)

    def inference(self, input_image_path):

        self.input_path = input_image_path
        # define cmd arguments
        parser = argparse.ArgumentParser()
        # parser.add_argument('--input-path', type=str, default="./test_image1", help='path of input images')
        parser.add_argument('--output-path', type=str, default="test_image_result", help='path of output images')
        parser.add_argument('--ckpt-path', type=str,
                            # default="../../../pretrained/modnet_custom_portrait_matting_25_th.ckpt",
                            default="pretrained/modnet_photographic_portrait_matting.ckpt",  # 效果最好
                            # default="../../../pretrained/modnet_epoch_299.pth",
                            help='path of pre-trained MODNet')
        args = parser.parse_args()

        # check input arguments
        # if not os.path.exists(args.input_path):
        #     print('Cannot find input path: {0}'.format(args.input_path))
        #     exit()
        # if not os.path.exists(args.output_path):
        #     print('Cannot find output path: {0}'.format(args.output_path))
        #     exit()
        if not os.path.exists(args.ckpt_path):
            print('Cannot find ckpt path: {0}'.format(args.ckpt_path))
            exit()
        # define hyper-parameters
        ref_size = 512

        # define image to tensor transform
        im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        # create MODNet and load the pre-trained ckpt
        modnet = MODNet(backbone_pretrained=False)
        modnet = nn.DataParallel(modnet)

        if torch.cuda.is_available():
            modnet = modnet.cuda()
            weights = torch.load(args.ckpt_path)
        else:
            weights = torch.load(args.ckpt_path, map_location=torch.device('cpu'))
        # modnet.load_state_dict(weights['model_state_dict'])
        modnet.load_state_dict(weights)
        modnet.eval()

        # read image
        im = Image.open(self.input_path)

        # unify image channels to 3
        # image = np.array(im)
        im = np.asarray(im)

        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]
        im_RGB = im
        # convert image to PyTorch tensor
        im = Image.fromarray(im)
        im = im_transform(im)

        # add mini-batch dim
        im = im[None, :, :, :]

        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

        # inference
        _, _, matte = modnet(im.cuda() if torch.cuda.is_available() else im, True)

        # resize and save matte
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
        matte = matte[0][0].data.cpu().numpy()

        matte_name = input_image_path.split('/')[-1]

        Image.fromarray(((matte * 255).astype('uint8')), mode='L').save(os.path.join(args.output_path, matte_name))

        image = Image.open(input_image_path)
        matte = Image.open(os.path.join(args.output_path, matte_name))

        w, h = image.width, image.height
        # w, h = image.shape[1], image.shape[0]

        # obtain predicted foreground
        image = np.asarray(image)
        if len(image.shape) == 2:
            image = image[:, :, None]
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.shape[2] == 4:
            image = image[:, :, 0:3]
        matte = np.asarray(matte)
        matte = np.repeat(np.asarray(matte)[:, :, None], 3, axis=2) / 255
        foreground = image * matte + np.full(image.shape, 255) * (1 - matte)

        fore_pil = Image.fromarray(np.uint8(foreground))
        fore_pil.save(os.path.join(args.output_path, matte_name.split('.')[0] + '.png'))

        fore2 = QtGui.QPixmap(os.path.join(args.output_path, matte_name.split('.')[0] + '.png')).scaled(
            self.input.width(), self.input.height())
        self.output.setPixmap(fore2)

    # 可视化
    def combined_display(self, image, matte):
        # calculate display resolution
        w, h = image.width, image.height
        rw, rh = 800, int(h * 800 / (3 * w))

        # obtain predicted foreground
        image = np.asarray(image)
        if len(image.shape) == 2:
            image = image[:, :, None]
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.shape[2] == 4:
            image = image[:, :, 0:3]
        matte = np.asarray(matte)
        matte = np.repeat(np.asarray(matte)[:, :, None], 3, axis=2) / 255
        foreground = image * matte + np.full(image.shape, 255) * (1 - matte)

        # combine image, foreground, and alpha into one line
        # combined = np.concatenate((image, foreground, matte * 255), axis=1)
        # combined = Image.fromarray(np.uint8(combined)).resize((rw, rh))
        fore = Image.fromarray(np.uint8(foreground))
        # return combined
        return fore

    def msg(self):

        return


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # inference images
    # 创建窗口
    w = MyWindow()

    # 展示窗口
    w.show()



    # 程序进行循环等待状态
    app.exec()
