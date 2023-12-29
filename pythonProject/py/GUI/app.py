from PySide6.QtWidgets import QApplication, QMainWindow,QPushButton,QLabel
from PySide6.QtCore import QSize
from PySide6.QtGui import QFont
from menu_zhiwen import Ui_Form_menu_zhiwen
from menu_renlian import Ui_Form_menu_renlian
import os

class Mywidow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setFixedSize(QSize(400, 300))
        btn1=QPushButton('指纹识别',self)
        btn1.setGeometry(260,60,110,60)
        font = QFont()
        font.setPointSize(18)
        btn1.setFont(font)
        btn1.clicked.connect(self.zhiwen)

        btn2=QPushButton('人脸识别',self)
        btn2.setGeometry(260, 150, 110, 60)
        font = QFont()
        font.setPointSize(18)
        btn2.setFont(font)
        btn2.clicked.connect(self.renlian)

        lab1=QLabel('指纹与人脸识别系统',self )
        lab1.setGeometry(20, 100, 220, 60)
        font = QFont()
        font.setPointSize(18)
        lab1.setFont(font)


    def zhiwen(self):
        self.subwindow=Subwindow2()
        self.subwindow.show()

    def renlian(self):
        self.subwindow=Subwindow1()
        self.subwindow.show()



class Subwindow1(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form_menu_renlian()
        self.ui.setupUi(self)

class Subwindow2(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form_menu_zhiwen()
        self.ui.setupUi(self)

if __name__ == '__main__':
    app = QApplication([])
    window = Mywidow()
    window.show()
    app.exec()


