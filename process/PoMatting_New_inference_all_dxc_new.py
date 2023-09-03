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
from PyQt5.QtCore import pyqtSignal, QThread
from PIL.ImageQt import toqpixmap
from threading import Thread
import time

from PyQt5.QtGui import *


#https://blog.csdn.net/Wang_Jiankun/article/details/81328499?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166660066916782428616042%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fspecialcolumn.%2522%257D&request_id=166660066916782428616042&biz_id=&utm_medium=distribute.pc_search_result.none-task-special_column-2~specialcolumn~first_rank_ecpm_v1~column_rank-5-9280833-null-null.pc_column&utm_term=pyqt5%E8%8F%9C%E9%B8%9F%E6%95%99%E7%A8%8B&spm=1018.2226.3001.4417

# 创建视频播放的线程类：
class MyThread(QThread):
    # 自定义信号
    #show_signal = pyqtSignal(QImage)

    # 构造函数，接受参数
    def __init__(self, file_name, browser):
        QThread.__init__(self)
        #self.video_name = file_name
        self.browser = browser
        self.video_name = file_name

    # 重写run()方法
    def run(self):
        # 实例化一个读取视频对象
        cap = cv2.VideoCapture(self.video_name)
        time.sleep(0.08)
        while cap.isOpened():
            # 读取视频帧
            ret, frame = cap.read()
            # 获取视频的帧数
            fps = cap.get(cv2.CAP_PROP_FPS)

            if ret:
                # 转换图片格式
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                qt_image = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], QImage.Format_RGB888)
                #p = qt_image.scaled(640, 480, Qt.KeepAspectRatio)
                # 发射信号
                #self.show_signal.emit(p)
                time.sleep(1 / fps)
                self.browser.setPixmap(QPixmap.fromImage(qt_image))
            # else:
            #     image = QImage('E:/11.jpg')
            #     self.browser.ui.lb_show.setPixmap(QPixmap.fromImage(image))
            #     self.browser.ui.lb_show.adjustSize()
            #     print('播放结束')
            #     break


# # 创建视频播放的线程类：
# class iVideo(QThread):
#     # 自定义信号
#     show_signal = pyqtSignal(QImage)
#
#     # 构造函数，接受参数
#     def __init__(self, file_name, browser):
#         QThread.__init__(self)
#         self.video_name = file_name
#         self.browser = browser
#
#     # 重写run()方法
#     def run(self):
#         # 实例化一个读取视频对象
#         cap = cv2.VideoCapture(self.video_name)
#
#         while cap.isOpened():
#             # 读取视频帧
#             ret, frame = cap.read()
#             # 获取视频的帧数
#             fps = cap.get(cv2.CAP_PROP_FPS)
#
#             if ret:
#                 # 转换图片格式
#                 rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 qt_image = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], QImage.Format_RGB888)
#                 p = qt_image.scaled(640, 480, Qt.KeepAspectRatio)
#                 # 发射信号
#                 self.show_signal.emit(p)
#                 time.sleep(1 / fps)
#             # else:
#             #     image = QImage('E:/11.jpg')
#             #     self.browser.ui.lb_show.setPixmap(QPixmap.fromImage(image))
#             #     self.browser.ui.lb_show.adjustSize()
#             #     print('播放结束')
#             #     break
#
# # 创建视频播放的线程类：
# class oVideo(QThread):
#     # 自定义信号
#     show_signal = pyqtSignal(QImage)
#
#     # 构造函数，接受参数
#     def __init__(self, file_name, browser):
#         QThread.__init__(self)
#         self.video_name = file_name
#         self.browser = browser
#
#     # 重写run()方法
#     def run(self):
#         # 实例化一个读取视频对象
#         cap = cv2.VideoCapture(self.video_name)
#
#         while cap.isOpened():
#             # 读取视频帧
#             ret, frame = cap.read()
#             # 获取视频的帧数
#             fps = cap.get(cv2.CAP_PROP_FPS)
#
#             if ret:
#                 # 转换图片格式
#                 rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 qt_image = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], QImage.Format_RGB888)
#                 p = qt_image.scaled(640, 480, Qt.KeepAspectRatio)
#                 # 发射信号
#                 self.show_signal.emit(p)
#                 time.sleep(1 / fps)
#             # else:
#             #     image = QImage('E:/11.jpg')
#             #     self.browser.ui.lb_show.setPixmap(QPixmap.fromImage(image))
#             #     self.browser.ui.lb_show.adjustSize()
#             #     print('播放结束')
#             #     break


# class iVideo(QThread):
#     def __init__(self, str, input):
#         super().__init__()
#         self.vInPath = str
#         self.input = input
#
#     def run(self):
#         if self.vInPath != "":
#             self.cap_in = cv2.VideoCapture(self.vInPath)
#             time.sleep(0.04)
#             self.frameRate = 20
#             # # 视频流设置 参数详见https://github.com/opencv/opencv/blob/master/modules/videoio/include/opencv2/videoio.hpp
#             self.cap_in.set(cv2.CAP_PROP_FPS, self.frameRate)  # 帧率 帧 / 秒   https: // www.javaroad.cn / questions / 288491
#
#             while self.cap_in.isOpened():
#                 success, frame = self.cap_in.read()
#                 # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 # frame = cv2.resize(frame,(self.input.height(),self.input.width()))
#
#                 if success == False:
#                     continue
#                 # img = QImage(frame.data, self.input.width(), self.input.height(), QImage.Format_RGB888)
#                 img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
#                 # mywindow.SetPic(a)
#                 self.input.setPixmap(QPixmap.fromImage(img))
#
#                 #if cv2.waitKey(100) & 0xFF == ord('q'):
#                     #break
#                 #cv2.waitKey(int(100 / self.frameRate))
#
#             # ret, image = self.cap_in.read()
#             # if ret:
#             #     if len(image.shape) == 3:
#             #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             #         vedio_img = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
#             #     elif len(image.shape) == 1:
#             #         vedio_img = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_Indexed8)
#             #     else:
#             #         vedio_img = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
#             #
#             #     self.output.setPixmap(QPixmap(vedio_img))
#             #     self.output.setScaledContents(True)  # 自适应窗口
#             # else:
#             #     self.cap_in.release()
#         #pass
#
# class oVideo(QThread):
#     def __init__(self, str, output):
#         super().__init__()
#         self.vInPath = str
#         self.output = output
#     def run(self):
#         if self.vInPath != "":
#             Path = self.vInPath.strip().rsplit("/", 2)
#             outName = Path[2].split(".")[0] + '_fg' + '.mp4'
#             self.vOutPath = os.path.join(Path[0], 'output_video', outName)
#
#             self.cap_out = cv2.VideoCapture(self.vOutPath)
#             time.sleep(0.04)
#             self.frameRate = 20
#             # # 视频流设置 参数详见https://github.com/opencv/opencv/blob/master/modules/videoio/include/opencv2/videoio.hpp
#             self.cap_out.set(cv2.CAP_PROP_FPS, self.frameRate)  # 帧率 帧 / 秒   https: // www.javaroad.cn / questions / 288491
#
#             while self.cap_out.isOpened():
#                 success, frame = self.cap_out.read()
#                 # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 # frame = cv2.resize(frame,(self.input.height(),self.input.width()))
#
#                 if success == False:
#                     continue
#                 # img = QImage(frame.data, self.input.width(), self.input.height(), QImage.Format_RGB888)
#                 img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
#                 # mywindow.SetPic(a)
#                 self.output.setPixmap(QPixmap.fromImage(img))
#
#                 # if cv2.waitKey(100) & 0xFF == ord('q'):
#                 #     break
#             # ret, image = self.cap_out.read()
#             # if ret:
#             #     if len(image.shape) == 3:
#             #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             #         vedio_img = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
#             #     elif len(image.shape) == 1:
#             #         vedio_img = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_Indexed8)
#             #     else:
#             #         vedio_img = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
#             #
#             #     self.output.setPixmap(QPixmap(vedio_img))
#             #     self.output.setScaledContents(True)  # 自适应窗口
#             # else:
#             #     self.cap_out.release()
#
#         #pass


class MyWindow(QMainWindow):
    displayEmitApp = pyqtSignal(str)
    displayEmitApp2 = pyqtSignal()
    displayEmitApp3 = pyqtSignal()

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

        # th = Thread(target= self.inputvideo)  # https://www.bbsmax.com/A/kmzLkAjKdG/
        # th.start()
        # time.sleep(0.01)

        # th1 = Thread(target=self.inputvideo)  # https://www.bbsmax.com/A/kmzLkAjKdG/
        # th2 = Thread(target=self.outputvideo)  # https://www.bbsmax.com/A/kmzLkAjKdG/


        # self.vIn = ""
        # self.th1 = iVideo(self.vIn, self)
        # self.th2 = oVideo(self.vIn, self)
        self.displayEmitApp2.connect(self.deal)

        # th2 = Thread(target=self.outputvideo)  # https://www.bbsmax.com/A/kmzLkAjKdG/
        # time.sleep(0.01)
        #th2.start()
        #self.displayEmitApp2.connect(self.outputvideo)

        #self.displayEmitApp3.connect(self.deal)



    def deal(self):

        self.th1 = MyThread(self.vIn, self.input)
        self.th2 = MyThread(self.vOut, self.output)

        # self.th1.show_signal.connect(self.inputvideo)
        # self.th2.show_signal.connect(self.outputvideo)

        self.th1.start()
        self.th2.start()

        #time.sleep(0.01)
        # self.th1.start()
        # self.th2.start()
        # th = Thread(target= self.inputvideo)  # https://www.bbsmax.com/A/kmzLkAjKdG/
        # th.start()
        # time.sleep(0.01)

        #th1 = Thread(target=self.inputvideo)  # https://www.bbsmax.com/A/kmzLkAjKdG/


        #th2 = Thread(target=self.outputvideo)  # https://www.bbsmax.com/A/kmzLkAjKdG/
        # th1.start()
        # time.sleep(0.01)
        # th2.start()
        # time.sleep(0.01)
        #pass



    def displayPath(self):
        self.vIn, imgType = QFileDialog.getOpenFileName(self, "打开mp4", "", "*.mp4;;*.png;;All Files(*)")
        self.path_display.setText(self.vIn)
        Path = self.vIn.strip().rsplit("/", 2)
        outName = Path[2].split(".")[0] + '_fg' + '.mp4'
        self.vOut = os.path.join(Path[0], 'output_video', outName)
        self.displayEmitApp2.emit()
        #self.displayEmitApp3.emit()


    def inputvideo(self,img):

        self.input.setPixmap(QPixmap.fromImage(img))
        # th1 = iVideo(self.vIn, self.input)
        # th1.start()

        # self.vInPath = self.vIn
        # self.cap_in = cv2.VideoCapture(self.vInPath)
        # self.frameRate = 20
        # # # 视频流设置 参数详见https://github.com/opencv/opencv/blob/master/modules/videoio/include/opencv2/videoio.hpp
        # self.cap_in.set(cv2.CAP_PROP_FPS, self.frameRate) # 帧率 帧 / 秒   https: // www.javaroad.cn / questions / 288491
        #
        #
        # while self.cap_in.isOpened():
        #     success, frame = self.cap_in.read()
        #     #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     #frame = cv2.resize(frame,(self.input.height(),self.input.width()))
        #
        #     if success == False:
        #         continue
        #     #img = QImage(frame.data, self.input.width(), self.input.height(), QImage.Format_RGB888)
        #     img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        #     #mywindow.SetPic(a)
        #     self.input.setPixmap(QPixmap.fromImage(img))
        #
        #     #cv2.waitKey(int(1000 / self.frameRate))


    def outputvideo(self,img):

        self.output.setPixmap(QPixmap.fromImage(img))
        # th2 = oVideo(self.vIn, self.output)
        # th2.start()

        # Path = self.vIn.strip().rsplit("/", 2)
        # outName = Path[2].split(".")[0] + '_fg' + '.mp4'
        # self.vOutPath = os.path.join(Path[0], 'output_video', outName)
        #
        # self.cap_out = cv2.VideoCapture(self.vOutPath )
        # self.frameRate = 20
        # # # 视频流设置 参数详见https://github.com/opencv/opencv/blob/master/modules/videoio/include/opencv2/videoio.hpp
        # self.cap_out.set(cv2.CAP_PROP_FPS, self.frameRate)  # 帧率 帧 / 秒   https: // www.javaroad.cn / questions / 288491
        #
        # while self.cap_out.isOpened():
        #     success, frame = self.cap_out.read()
        #     # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     # frame = cv2.resize(frame,(self.input.height(),self.input.width()))
        #
        #     if success == False:
        #         continue
        #     # img = QImage(frame.data, self.input.width(), self.input.height(), QImage.Format_RGB888)
        #     img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        #     # mywindow.SetPic(a)
        #     self.output.setPixmap(QPixmap.fromImage(img))
        #     th2.start()
        #     time.sleep(0.01)
        #     #cv2.waitKey(int(1000 / self.frameRate))


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
