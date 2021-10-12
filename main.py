from math import fabs
import sys
import cv2

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

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
        self.comboBox.currentIndexChanged.connect(self.combo_box_func)
        self.registration_data = None

    def capture_button_func(self):
        self.registration_data = [int(self.ID.text()), self.name.text()]
        self.hide_menus()

    def remove_button_func(self):
        self.face_recognition.unregister_face(self.ID.text())
        self.hide_menus()

    def combo_box_func(self, i):
        combo_box_functions = {0: self.hide_menus, 1: self.register_item, 2:self.delete_item}
        combo_box_functions[i]()

    def delete_item(self):
        self.box_delete.show()
        self.box_register.hide() 
        self.recognition_on = False

    def register_item(self):
        self.box_register.show()
        self.box_delete.hide()
        self.recognition_on = False

    def hide_menus(self):
        self.box_delete.hide()
        self.box_register.hide()   
        self.comboBox.setCurrentText("")  
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
        processed_frame = self.face_recognition.process_frame(frame, self.recognition_on, self.registration_data)
        self.registration_data = None

        image = QImage(processed_frame, processed_frame.shape[1], processed_frame.shape[0], 
                       processed_frame.strides[0], QImage.Format_RGB888)
        self.Image.setPixmap(QPixmap.fromImage(image))

if (__name__ == '__main__'):

    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
