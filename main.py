import sys
import cv2
import argparse
import logging 

import PySide2
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from qt_material import apply_stylesheet

import qimage2ndarray

from qt_material import apply_stylesheet

from UI import Ui_MainWindow
from face_rec import FaceRecognition

class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self):

        QMainWindow.__init__(self)
        self.setupUi(self)
        self.setup_camera()
        self.face_recognition = FaceRecognition(0.7, self.frame_height, self.frame_width)        
        self.setWindowTitle("Face Recognition Demo")
        self.hide_menus()
        self.quitButton.clicked.connect(self.close)
        self.capture_button.clicked.connect(self.capture_button_func)
        self.remove_button.clicked.connect(self.remove_button_func)
        self.gpio_button.clicked.connect(self.gpio_button_func)
        self.mqtt_button.clicked.connect(self.mqtt_button_func)        
        self.menuBox.currentIndexChanged.connect(self.combo_box_func)
        self.stateBox.currentIndexChanged.connect(self.state_box_func)        
        self.registration_data = None
        self.MQTT_state = False
        self.GPIO_state = False

        qt_version = PySide2.QtCore.__version_info__
        if qt_version[0] < 5 or qt_version[1] < 12:
            print(f"Your QT version is {qt_version}, lower than 5.12 required for material design!")
            apply_material = False
        else: 
            apply_stylesheet(self, theme='dark_teal.xml')

        self.setWindowFlags(Qt.FramelessWindowHint)
        self.showFullScreen()

    def capture_button_func(self):
        self.registration_data = [int(self.ID.text()), self.name.text()]
        self.hide_menus()

    def remove_button_func(self):
        self.face_recognition.unregister_face(self.ID.text())
        self.hide_menus()

    def mqtt_button_func(self):

        if not self.MQTT_state:
            self.MQTT = AddonMQTT(self.address.text(), int(self.port.text()))

        else:
            del self.MQTT

        self.MQTT_state = not self.MQTT_state

    def gpio_button_func(self):
        if not self.GPIO_state:
            self.GPIO = AddonGPIO(int(self.pin.text()), self.active_state)

        else:
            del self.GPIO

        self.GPIO_state = not self.GPIO_state

    def combo_box_func(self, i):
        combo_box_functions = {0: self.hide_menus, 1: self.register_item,
                               2:self.delete_item, 3:self.conf_mqtt,
                               4:self.conf_gpio}
        self.hide_menus()
        combo_box_functions[i]()

    def state_box_func(self, i):
        self.active_state = int(i)

    def delete_item(self):
        self.box_delete.show()
        self.box_delete.move(80, 80)
        self.box_register.hide() 
        self.recognition_on = False

    def register_item(self):
        self.box_register.show()
        self.box_delete.hide()
        self.recognition_on = False

    def conf_mqtt(self):
        self.box_mqtt.show()
        self.box_mqtt.move(80, 80)        

    def conf_gpio(self):
        self.box_gpio.show()
        self.box_gpio.move(80, 80)

    def hide_menus(self):
        self.box_delete.hide()
        self.box_register.hide()
        self.box_mqtt.hide()   
        self.box_gpio.hide()

        self.menuBox.setCurrentText("")  
        self.recognition_on = True

    def setup_camera(self):
        self.capture = cv2.VideoCapture(0)
        self.frame_height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frame_width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)

        self.timer = QTimer()
        self.timer.timeout.connect(self.display_video_stream)
        self.timer.start(30)

    def display_video_stream(self):
        _, frame = self.capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        processed_frame, ids = self.face_recognition.process_frame(frame, self.recognition_on, self.registration_data)
        processed_frame = cv2.resize(processed_frame, (1280, 720))
        self.registration_data = None

        if len(ids) > 0 and self.MQTT_state:
            self.MQTT.send_data("face_rec/verified", 1)

        if len(ids) > 0 and self.GPIO_state:
            self.GPIO.control_gpio()

        image = qimage2ndarray.array2qimage(processed_frame)
        self.Image.setPixmap(QPixmap.fromImage(image))

if (__name__ == '__main__'):

    argparser = argparse.ArgumentParser(
        description='Run Face Recognition QT APP')

    argparser.add_argument(
        '--use_mqtt',
        action='store_true',
        default=False,
        help='whether to have MQTT addon')

    argparser.add_argument(
        '--use_gpio',
        action='store_true',
        default=False,
        help='whether to have GPIO adddon')

    args = argparser.parse_args()

    if args.use_mqtt:
        from addons.mqtt_addon import AddonMQTT

    if args.use_gpio:
        from addons.gpio_addon import AddonGPIO

    logging.basicConfig(level=logging.INFO)

    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())