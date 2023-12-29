# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'menu_zhiwen.ui'
##
## Created by: Qt User Interface Compiler version 6.6.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QPushButton, QSizePolicy, QWidget)
import os

import zhiwentest

class Ui_Form_menu_zhiwen(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(400, 300)
        self.pushButton = QPushButton(Form)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(50, 100, 101, 101))
        self.pushButton.clicked.connect(self.zhiwenluru)
        font = QFont()
        font.setPointSize(18)
        self.pushButton.setFont(font)
        self.pushButton_2 = QPushButton(Form)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setGeometry(QRect(230, 100, 101, 101))
        self.pushButton_2.clicked.connect(self.zhiwenshibie)
        font1 = QFont()
        font1.setPointSize(18)
        self.pushButton_2.setFont(font1)

        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi
    def zhiwenshibie(self):#指纹识别的传递函数
        zhiwentest.zhiwenshibie_main()

    def zhiwenluru(self):#指纹识别的传递函数
        None

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.pushButton.setText(QCoreApplication.translate("Form", u"\u6307\u7eb9\u5f55\u5165", None))
        self.pushButton_2.setText(QCoreApplication.translate("Form", u"\u6307\u7eb9\u8bc6\u522b", None))
    # retranslateUi

