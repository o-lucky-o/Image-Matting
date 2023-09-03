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

class MyWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):

        self.input_path = None

        # 窗口的大小
        self.resize(800, 500) #宽800(x), 高500(y)

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
        self.path_display.setGeometry(60, 50, 730, 20)  # 横着的x，竖着的y，长度，宽度
        #self.path_display.move(100, 50)
        # 字体设置
        self.path_display.setStyleSheet("QLabel{background:white;}"
                                        "QLabel{color:rgb(300,300,300,120);font-size:15px;font-weight:bold;font-family:宋体;}")

        # 纯文本
        self.input = QLabel(self)
        self.input.setText("")
        self.input.setGeometry(60, 80, 350, 400)  # * , * 长度(右)，宽度(左)
        # self.label.setFixedSize(int(410 * 2 / 3), int(410 * 2 / 3))
        #self.input.move(160, 100)
        self.input.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:15px;font-weight:bold;font-family:宋体;}")

        # 纯文本
        self.output = QLabel(self)
        self.output.setText("")
        self.output.setGeometry(440, 80, 350, 400)  # * , * 长度(右)，宽度(左)
        # self.label.setFixedSize(int(410 * 2 / 3), int(410 * 2 / 3))
        #self.output.move(160, 100)
        self.output.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:15px;font-weight:bold;font-family:宋体;}")


        # # 文本框
        # self.path_display = QLineEdit(self)
        i_open.triggered.connect(self.openimage)



    def openimage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")

        jpg = QtGui.QPixmap(imgName).scaled(self.input.width(), self.input.height())
        self.input.setPixmap(jpg)
        self.path_display.setText(imgName)
        self.inference(imgName)
        self.output.setPixmap()


    def inference(self,input_image_path):

        self.input_path = input_image_path


        return


    def msg(self):

        return


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 创建窗口
    w = MyWindow()

    # 展示窗口
    w.show()

    # 程序进行循环等待状态
    app.exec()
