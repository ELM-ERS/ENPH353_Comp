import roslib
import rospy
import sys
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from datetime import datetime

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi


class RobotViewApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(RobotViewApp, self).__init__()
        loadUi("./robot_view.ui", self)

        self.bridge = CvBridge()

        # TODO: figure out what exactly to subscribe to here.
        self.image_sub = rospy.Subscriber(
            "/B1/rrbot/camera1/image_raw", Image, self.cv_bridge_callback
        )

        self.recording_button.clicked.connect(self.SLOT_recording_button_handler)
        self.pause_button.clicked.connect(self.SLOT_pause_button_handler)
        self.save_images_button.clicked.connect(self.SLOT_save_images_button_handler)

        self.recording = False
        self.paused = False

    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(
            cv_img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888
        )
        return QtGui.QPixmap.fromImage(q_img)

    def cv_bridge_callback(self, data):
        try:
            self.raw_cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            if self.recording:
                self.save_current_image(prefix="recording/")
        except CvBridgeError as e:
            print(e)

        # print(self.raw_cv_image.shape)
        resized = cv2.resize(self.raw_cv_image, (400, 400))
        pixmap = self.convert_cv_to_pixmap(resized)
        # print(resized.shape)
        self.live_image_label.setPixmap(pixmap)

    def save_current_image(self, prefix=""):
        now = datetime.now()
        f = now.strftime("%Y_%m_%d_%H_%M_%S_%f.png")
        path_string = "./current_imgs/" + prefix + "img_" + f
        print(path_string)
        cv2.imwrite(path_string, self.raw_cv_image)

    def SLOT_recording_button_handler(self):
        self.recording = not self.recording

    def SLOT_pause_button_handler(self):
        self.paused = not self.paused

    def SLOT_save_images_button_handler(self):
        self.save_current_image()


if __name__ == "__main__":
    print(cv2.__version__)
    rospy.init_node("robot_view", anonymous=True)
    app = QtWidgets.QApplication(sys.argv)
    robotViewApp = RobotViewApp()
    robotViewApp.show()
    sys.exit(app.exec_())
