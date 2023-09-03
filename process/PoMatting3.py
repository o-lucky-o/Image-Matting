import sys
from PyQt5 import  QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog,QWidget, QLabel, QApplication, QLineEdit, QVBoxLayout,QGroupBox, QVBoxLayout, QHBoxLayout


class MyWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):

        # 窗口的大小
        self.resize(800, 500)

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
        path.setGeometry(20, 50, 50, 20)

        #字体设置
        # label.setStyleSheet("font-size:30px;color:red")
        # 设置中心内容显示
        # self.setCentralWidget(label)

        # 文本框
        path_display = QLineEdit(self)
        path_display.setGeometry(100, 50, 350, 20)  # 竖着的，横着的，长度，宽度

        i_open.triggered.connect(self.openimage)

        # self.labels = QLabel('file',self)
        # #self.labels.setText('file')                                                                                ')
        # self.labels.setGeometry(100, 50, 350, 200)
        # #self.labels.move(150, 60)

        self.label = QLabel(self)
        self.label.setText("lll")
        self.label.setFixedSize(int(410 * 2 / 3), int(410 * 2 / 3))
        self.label.move(160, 100)

        # self.labels = QLabel(self)
        # self.labels.setText("")
        # self.labels.move(150, 60)

    def openimage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        print(imgName)
        jpg = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg)
        # self.labels.setText(imgName)


    def msg(self):

        return


if __name__ == '__main__':
    app = QApplication(sys.argv)

    w = MyWindow()
    # 设置窗口标题

    # # 窗口的大小
    # w.resize(500, 500)
    # 展示窗口
    w.show()

    # 程序进行循环等待状态
    app.exec()
