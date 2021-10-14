# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'UI.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1280, 720)
        MainWindow.setWindowOpacity(1.000000000000000)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.comboBox = QComboBox(self.centralwidget)
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.setObjectName(u"comboBox")
        self.comboBox.setGeometry(QRect(20, 10, 491, 41))
        self.comboBox.setAutoFillBackground(False)
        self.comboBox.setEditable(False)
        self.quitButton = QPushButton(self.centralwidget)
        self.quitButton.setObjectName(u"quitButton")
        self.quitButton.setGeometry(QRect(870, 610, 401, 51))
        self.quitButton.setAutoFillBackground(False)
        self.Image = QLabel(self.centralwidget)
        self.Image.setObjectName(u"Image")
        self.Image.setGeometry(QRect(10, -1, 1271, 701))
        self.box_register = QWidget(self.centralwidget)
        self.box_register.setObjectName(u"box_register")
        self.box_register.setGeometry(QRect(80, 80, 280, 150))
        self.formLayout = QFormLayout(self.box_register)
        self.formLayout.setObjectName(u"formLayout")
        self.ID_label = QLabel(self.box_register)
        self.ID_label.setObjectName(u"ID_label")

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.ID_label)

        self.ID = QLineEdit(self.box_register)
        self.ID.setObjectName(u"ID")

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.ID)

        self.name_label = QLabel(self.box_register)
        self.name_label.setObjectName(u"name_label")

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.name_label)

        self.name = QLineEdit(self.box_register)
        self.name.setObjectName(u"name")

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.name)

        self.capture_button = QPushButton(self.box_register)
        self.capture_button.setObjectName(u"capture_button")

        self.formLayout.setWidget(2, QFormLayout.SpanningRole, self.capture_button)

        self.box_delete = QWidget(self.centralwidget)
        self.box_delete.setObjectName(u"box_delete")
        self.box_delete.setGeometry(QRect(80, 230, 271, 121))
        self.formLayout_2 = QFormLayout(self.box_delete)
        self.formLayout_2.setObjectName(u"formLayout_2")
        self.ID_label_2 = QLabel(self.box_delete)
        self.ID_label_2.setObjectName(u"ID_label_2")

        self.formLayout_2.setWidget(0, QFormLayout.LabelRole, self.ID_label_2)

        self.ID_2 = QLineEdit(self.box_delete)
        self.ID_2.setObjectName(u"ID_2")

        self.formLayout_2.setWidget(0, QFormLayout.FieldRole, self.ID_2)

        self.remove_button = QPushButton(self.box_delete)
        self.remove_button.setObjectName(u"remove_button")

        self.formLayout_2.setWidget(1, QFormLayout.FieldRole, self.remove_button)

        MainWindow.setCentralWidget(self.centralwidget)
        self.Image.raise_()
        self.comboBox.raise_()
        self.quitButton.raise_()
        self.box_register.raise_()
        self.box_delete.raise_()
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1280, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.comboBox.setItemText(0, "")
        self.comboBox.setItemText(1, QCoreApplication.translate("MainWindow", u"Register New User", None))
        self.comboBox.setItemText(2, QCoreApplication.translate("MainWindow", u"Delete User ID", None))

        self.comboBox.setCurrentText("")
        self.quitButton.setText(QCoreApplication.translate("MainWindow", u"Quit", None))
        self.Image.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.ID_label.setText(QCoreApplication.translate("MainWindow", u"ID", None))
        self.ID.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.name_label.setText(QCoreApplication.translate("MainWindow", u"Name", None))
        self.name.setText(QCoreApplication.translate("MainWindow", u"John Doe", None))
        self.capture_button.setText(QCoreApplication.translate("MainWindow", u"Capture", None))
        self.ID_label_2.setText(QCoreApplication.translate("MainWindow", u"ID", None))
        self.ID_2.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.remove_button.setText(QCoreApplication.translate("MainWindow", u"Remove User ID", None))
    # retranslateUi

