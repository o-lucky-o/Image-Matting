import sys

from PyQt5.QtWidgets import QMainWindow, QLabel, QApplication, QLineEdit, QVBoxLayout,QGroupBox, QVBoxLayout, QHBoxLayout


class MyWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):

        # 窗口的大小
        self.resize(800, 500)

        # 设置窗口标题
        self.setWindowTitle("PoMatting")

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


        # 最外层的垂直布局，包含两部分：文件路径和处理
        container = QVBoxLayout()

        # -----创建第1个组，添加多个组件-----
        path_box = QGroupBox("")
        path_layout = QHBoxLayout()
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

        path_layout.addWidget(path)
        path_layout.addWidget(path_display)
        path_box.setLayout(path_layout)

        # -----创建第2个组，添加多个组件-----
        path_box = QGroupBox("")
        path_layout = QHBoxLayout()
        container.addWidget(path_box)
        # 设置窗口显示的内容是最外层容器
        self.setLayout(container)


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
