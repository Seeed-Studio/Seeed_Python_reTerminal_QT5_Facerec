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
        self.menuBox = QComboBox(self.centralwidget)
        self.menuBox.addItem("")
        self.menuBox.addItem("")
        self.menuBox.addItem("")
        self.menuBox.addItem("")
        self.menuBox.addItem("")
        self.menuBox.setObjectName(u"menuBox")
        self.menuBox.setGeometry(QRect(20, 10, 491, 41))
        self.menuBox.setAutoFillBackground(False)
        self.menuBox.setEditable(False)
        self.quitButton = QPushButton(self.centralwidget)
        self.quitButton.setObjectName(u"quitButton")
        self.quitButton.setGeometry(QRect(870, 610, 401, 51))
        self.quitButton.setAutoFillBackground(False)
        self.Image = QLabel(self.centralwidget)
        self.Image.setObjectName(u"Image")
        self.Image.setGeometry(QRect(0, 0, 1280, 720))
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

        self.box_mqtt = QWidget(self.centralwidget)
        self.box_mqtt.setObjectName(u"box_mqtt")
        self.box_mqtt.setGeometry(QRect(80, 360, 271, 131))
        self.formLayout_4 = QFormLayout(self.box_mqtt)
        self.formLayout_4.setObjectName(u"formLayout_4")
        self.address_label = QLabel(self.box_mqtt)
        self.address_label.setObjectName(u"address_label")

        self.formLayout_4.setWidget(0, QFormLayout.LabelRole, self.address_label)

        self.address = QLineEdit(self.box_mqtt)
        self.address.setObjectName(u"address")

        self.formLayout_4.setWidget(0, QFormLayout.FieldRole, self.address)

        self.port_label = QLabel(self.box_mqtt)
        self.port_label.setObjectName(u"port_label")

        self.formLayout_4.setWidget(1, QFormLayout.LabelRole, self.port_label)

        self.port = QLineEdit(self.box_mqtt)
        self.port.setObjectName(u"port")

        self.formLayout_4.setWidget(1, QFormLayout.FieldRole, self.port)

        self.mqtt_button = QPushButton(self.box_mqtt)
        self.mqtt_button.setObjectName(u"mqtt_button")

        self.formLayout_4.setWidget(2, QFormLayout.SpanningRole, self.mqtt_button)

        self.box_gpio = QWidget(self.centralwidget)
        self.box_gpio.setObjectName(u"box_gpio")
        self.box_gpio.setGeometry(QRect(80, 520, 271, 131))
        self.formLayout_5 = QFormLayout(self.box_gpio)
        self.formLayout_5.setObjectName(u"formLayout_5")
        self.pin_label = QLabel(self.box_gpio)
        self.pin_label.setObjectName(u"pin_label")

        self.formLayout_5.setWidget(0, QFormLayout.LabelRole, self.pin_label)

        self.pin = QLineEdit(self.box_gpio)
        self.pin.setObjectName(u"pin")

        self.formLayout_5.setWidget(0, QFormLayout.FieldRole, self.pin)

        self.stateBox = QComboBox(self.box_gpio)
        self.stateBox.addItem("")
        self.stateBox.addItem("")
        self.stateBox.setObjectName(u"stateBox")

        self.formLayout_5.setWidget(1, QFormLayout.FieldRole, self.stateBox)

        self.gpio_button = QPushButton(self.box_gpio)
        self.gpio_button.setObjectName(u"gpio_button")

        self.formLayout_5.setWidget(2, QFormLayout.SpanningRole, self.gpio_button)

        self.state_label = QLabel(self.box_gpio)
        self.state_label.setObjectName(u"state_label")

        self.formLayout_5.setWidget(1, QFormLayout.LabelRole, self.state_label)

        MainWindow.setCentralWidget(self.centralwidget)
        self.Image.raise_()
        self.menuBox.raise_()
        self.quitButton.raise_()
        self.box_register.raise_()
        self.box_delete.raise_()
        self.address_label.raise_()
        self.port_label.raise_()
        self.mqtt_button.raise_()
        self.address.raise_()
        self.port.raise_()
        self.pin_label.raise_()
        self.state_label.raise_()
        self.stateBox.raise_()
        self.pin.raise_()
        self.gpio_button.raise_()
        self.box_mqtt.raise_()
        self.box_gpio.raise_()
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
        self.menuBox.setItemText(0, "")
        self.menuBox.setItemText(1, QCoreApplication.translate("MainWindow", u"Register New User", None))
        self.menuBox.setItemText(2, QCoreApplication.translate("MainWindow", u"Delete User ID", None))
        self.menuBox.setItemText(3, QCoreApplication.translate("MainWindow", u"Configure MQTT", None))
        self.menuBox.setItemText(4, QCoreApplication.translate("MainWindow", u"Configure GPIO", None))

        self.menuBox.setCurrentText("")
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
        self.address_label.setText(QCoreApplication.translate("MainWindow", u"Address", None))
        self.address.setText(QCoreApplication.translate("MainWindow", u"localhost", None))
        self.port_label.setText(QCoreApplication.translate("MainWindow", u"Port", None))
        self.port.setText(QCoreApplication.translate("MainWindow", u"1883", None))
        self.mqtt_button.setText(QCoreApplication.translate("MainWindow", u"Start/Stop MQTT", None))
        self.pin_label.setText(QCoreApplication.translate("MainWindow", u"Pin", None))
        self.pin.setText(QCoreApplication.translate("MainWindow", u"17", None))
        self.stateBox.setItemText(0, QCoreApplication.translate("MainWindow", u"HIGH", None))
        self.stateBox.setItemText(1, QCoreApplication.translate("MainWindow", u"LOW", None))

        self.gpio_button.setText(QCoreApplication.translate("MainWindow", u"Start/Stop GPIO control", None))
        self.state_label.setText(QCoreApplication.translate("MainWindow", u"Active state", None))
    # retranslateUi

