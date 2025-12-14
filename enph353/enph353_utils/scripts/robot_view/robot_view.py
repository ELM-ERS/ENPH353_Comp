import roslib
import rospy
import sys
import os
import csv
from pathlib import Path
import numpy
import cv2

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

from datetime import datetime

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
import torch
import numpy as np
from sklearn.cluster import DBSCAN

from helpers import HotPixelAnalyzer, CluePublisher, ClueTracker

from il_data_recorder import ImageVectorRecorder
from cnn_driver_inference import CNNInference

CLUSTER_EPS = 0.12

SECTION_2_FRAME_COUNT = 10


class StateMachine:
    def __init__(self, m_percent_thresh=1.0):
        self.state = 0
        self.m_triggered = False
        self.m_percent_thresh = m_percent_thresh
        self.changed = False

    def update_state(self, m_percent, driver_forward_val=0):
        # If magenta percentage is triggered and it's not already triggered, increment state.
        if m_percent > self.m_percent_thresh and not self.m_triggered:
            self.m_triggered = True
            self.state += 1
            self.changed = True

        # Reset magenta trigger if below lower threshold.
        if m_percent < 0.01 and self.m_triggered:
            self.m_triggered = False
        return self.state

    def get_state(self):
        return self.state

    def _changed(self):
        if self.changed:
            self.changed = False
            return True


class YodaThresholdDetector:
    def __init__(self, threshold=0.01, min_frames_before=20, min_frames_after=6):
        self.threshold = threshold
        self.threshold_tripped = False
        self.min_val = None

        self.min_frames_before = min_frames_before
        self.min_frames_after = min_frames_after
        self.frames_before = 0
        self.frames_after = 0

    # Returns true if tripped.
    def new_value(self, val):
        if self.frames_before < self.min_frames_before:
            # print("YODA FRAME COUNT BEFORE: ", self.frames_before)
            self.frames_before += 1
            print("NOT ENOUGH FRAMES BEFORE YET!")
            return False

        # print("YODA VAL: ", val)
        if self.min_val is None:
            self.min_val = val
            return False

        if val - self.min_val >= self.threshold:
            self.threshold_tripped = True

        self.min_val = min(val, self.min_val)

        # print("YODA MIN VAL: ", self.min_val)
        # print("YODA TRIPPED: ", self.threshold_tripped)
        # print("\n\n")

        if self.threshold_tripped and self.frames_after < self.min_frames_after:
            # print("YODA FRAME COUNT BEFORE: ", self.frames_after)
            self.frames_after += 1
            print("NOT ENOUGH FRAMES AFTER YET!")
            return False

        return self.threshold_tripped


class Counter:
    def __init__(self, count=10):
        self._count = count
        self.val = 0
        self.done = False

    def count(self):
        if not self.done:
            self.val += 1

        if self.val > self._count:
            self.done = True

        return self.done


TEAM_NAME = "ELMERS"
PASSWORD = "password"

CLUE_IDS = {
    "SIZE": 1,
    "VICTIM": 2,
    "CRIME": 3,
    "TIME": 4,
    "PLACE": 5,
    "MOTIVE": 6,
    "WEAPON": 7,
    "BANDIT": 8,
}

FRAMES_PER_CLUE_SUBMISSION = 200


def publish_clues(publisher, clues):
    print("PUBLISHING CLUES!")
    print("PUBLISHING CLUES!")
    for clue_type, clue in clues.items():
        publisher.publish(f"{TEAM_NAME},{PASSWORD},{CLUE_IDS[clue_type]},{clue}")


def publish_timer_start(publisher):
    print("PUBLISHING TIMER START!")
    publisher.publish(f"{TEAM_NAME},{PASSWORD},0,NA")


def publish_timer_stop(publisher):
    print("PUBLISHING TIMER STOP!")
    publisher.publish(f"{TEAM_NAME},{PASSWORD},-1,NA")


class RobotViewApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(RobotViewApp, self).__init__()
        loadUi("./robot_view.ui", self)

        # self.yolo_model = YOLO("./main_sign_model.pt")
        self.yolo_model = YOLO("./main_sign_model.torchscript")

        # self.driver = CNNInference("./cnn_driver_loss_0_44.pth")
        self.driver = {
            "Full": CNNInference("./model_backups/full/cnn_driver_long_overnight.pth"),
            "Sec 1": CNNInference("./model_backups/section1/test1v4.pth"),
            # "Sec 2": CNNInference("./model_backups/section2/test2.pth"),
            "Sec 2": CNNInference("./model_backups/section2/test2_COMP_DAYv1.pth"),
            "Sec 3": CNNInference("./model_backups/section3/test2v2.pth"),
            "Sec 4": CNNInference("./model_backups/section4/test1.pth"),
        }

        self.all_clues_published = False

        self.frames_since_last_clue_publish = 0

        self.m_detector = HotPixelAnalyzer()
        self.m_detector.init(200)
        self.state_machine = StateMachine()
        self.yoda_threshold_detector = YodaThresholdDetector()
        self.have_we_seen_yoda = False
        self.section_2_frame_counter = Counter(SECTION_2_FRAME_COUNT)
        self.clue_publisher = ClueTracker()
        # self.clue_publisher = CluePublisher()

        # self.yolo_model = DetectionModel("yolo12m.yaml")
        # ckpt = torch.load("./main_sign_model.pt", map_location="cuda")
        # self.yolo_model.load_state_dict(ckpt["model"])

        self.bridge = CvBridge()

        self.latest_clue = ("", "")

        self.il_data_recorder = None

        # TODO: figure out what exactly to subscribe to here.
        self.image_sub = rospy.Subscriber(
            "/B1/rrbot/camera1/image_raw", Image, self.cv_bridge_callback
        )

        self.started = False
        # self.finished = False

        self.cmd_vel_sub = rospy.Subscriber("/B1/cmd_vel", Twist, self.cmd_vel_callback)

        self.cmd_vel_pub = rospy.Publisher("/B1/cmd_vel", Twist, queue_size=10)

        self.score_publisher = rospy.Publisher("/score_tracker", String, queue_size=10)

        self.il_recording_button.clicked.connect(self.SLOT_il_recording_button_handler)
        self.sign_reading_button.clicked.connect(self.SLOT_sign_reading_button_handler)
        self.autopilot_button.clicked.connect(self.SLOT_autopilot_button_handler)
        self.reset_state_button.clicked.connect(self.SLOT_reset_state_and_clues)

        # Autopilot combo box
        self.autopilot_section_combo_box.addItem("Full")
        self.autopilot_section_combo_box.addItem("Sec 1")
        self.autopilot_section_combo_box.addItem("Sec 2")
        self.autopilot_section_combo_box.addItem("Sec 3")
        self.autopilot_section_combo_box.addItem("Sec 4")

        self.full_auto_combo_box.addItem("On")
        self.full_auto_combo_box.addItem("Off")
        self.full_auto_combo_box.setCurrentIndex(1)

        self.current_linear_value = 0
        self.current_angular_value = 0

        self.il_data_counter = 0
        self.il_data_prefix = "./no_prefix"

        self.il_recording = False
        self.sign_reading = True

        self.autopilot = False

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

            m_percent = self.m_detector.analyze_image(self.raw_cv_image)

            self.magenta_percent_str.setText(f"{m_percent:.2f}")

            state = self.state_machine.update_state(m_percent)

            self.state_str.setText(f"{state}")

            # STATE MACHINE AUTOPILOT;
            if self.full_auto_combo_box.currentText() == "On":
                # Set autopilot to true and update button text.
                self.autopilot = False
                self.SLOT_autopilot_button_handler()

                # Set current model based on state machine.
                state = self.state_machine.get_state()
                if state <= 1:
                    self.autopilot_section_combo_box.setCurrentIndex(1)
                elif state <= 4:
                    self.autopilot_section_combo_box.setCurrentIndex(state)

            elif self.full_auto_combo_box.currentText() == "Off":
                if self.state_machine._changed():
                    self.autopilot = True
                    self.SLOT_autopilot_button_handler()

                    self.stop_robot_twist()

            if self.sign_reading:
                self.annotated_image = self.draw_yolo12_detections_bgr(
                    self.raw_cv_image
                )
            if self.il_recording:
                self.record_il()
                # self.save_current_image(prefix="recording/")

            self.driver_output = self.driver[
                self.autopilot_section_combo_box.currentText()
            ].predict(self.raw_cv_image)

            self.driver_output_str.setText(
                "DRIVER OUTPUT: "
                + f"{self.driver_output[0]:.2f}, {self.driver_output[1]:.2f}"
            )

            self.frames_since_last_clue_publish += 1

            if self.frames_since_last_clue_publish >= FRAMES_PER_CLUE_SUBMISSION:
                freq_clues = self.clue_publisher.get_frequent_clues()
                self.frames_since_last_clue_publish = 0
                publish_clues(self.score_publisher, freq_clues)

            # print("DRIVER OUTPUT: ", self.driver_output)

            # If autopilot is enabled, drive the bot;
            # Special case for waiting for Yoda.
            if self.autopilot:
                # Wait until the robot gets excited because Yoda is here!
                if self.full_auto_combo_box.currentText() == "On":
                    if not self.started:
                        publish_timer_start(self.score_publisher)
                        self.started = True

                    if "BANDIT" in self.clue_publisher.get_first_clues():
                        self.stop_robot_twist()
                        if not self.all_clues_published:
                            publish_timer_stop(self.score_publisher)
                            freq_clues = self.clue_publisher.get_frequent_clues()
                            print("FREQUENT CLUES: ", freq_clues)
                            self.all_clues_published = True
                            publish_clues(self.score_publisher, freq_clues)
                            print(
                                "FIRST CLUES: ", self.clue_publisher.get_first_clues()
                            )

                            print(
                                "FREQUENT CLUES: ",
                                freq_clues,
                            )

                    elif state == 2:
                        if self.section_2_frame_counter.count():
                            self.publish_driver_twist()
                        else:
                            self.stop_robot_twist()

                    elif state == 3:
                        yoda_detected = self.yoda_threshold_detector.new_value(
                            self.driver_output[0]
                        )

                        if not self.have_we_seen_yoda and yoda_detected:
                            self.have_we_seen_yoda = True
                            print("SAW YODA FOR THE FIRST TIME!")
                            self.yoda_threshold_detector = YodaThresholdDetector(
                                min_frames_before=50, min_frames_after=14
                            )
                            self.stop_robot_twist()

                        elif yoda_detected:
                            print("SAW YODA AGAIN! FOLLOWING")
                            self.publish_driver_twist()

                        else:
                            self.stop_robot_twist()
                    else:
                        self.publish_driver_twist()

                else:
                    self.publish_driver_twist()

        except CvBridgeError as e:
            print(e)

        img = self.raw_cv_image
        if self.sign_reading:
            img = self.annotated_image

        # print(self.raw_cv_image.shape)
        resized = cv2.resize(img, (400, 400))
        pixmap = self.convert_cv_to_pixmap(resized)
        # print(resized.shape)
        self.live_image_label.setPixmap(pixmap)

    def publish_driver_twist(self):
        twist = Twist()
        twist.linear.x = self.driver_output[0]
        twist.angular.z = self.driver_output[1]
        self.cmd_vel_pub.publish(twist)

    def stop_robot_twist(self):
        twist = Twist()
        twist.linear.x = 0
        twist.angular.z = 0
        self.cmd_vel_pub.publish(twist)

    def record_il(self):
        # Output data to file.
        if self.il_data_recorder:
            self.il_data_recorder.add_sample(
                self.raw_cv_image,
                (self.current_linear_value, self.current_angular_value),
            )

    def cmd_vel_callback(self, msg: Twist):
        self.current_linear_value = msg.linear.x
        self.current_angular_value = msg.angular.z

        # print(msg)
        # print("linear: {}", self.current_linear_value)
        # print("angular: {}", self.current_angular_value)

    def save_current_image(self, prefix=""):
        now = datetime.now()
        f = now.strftime("%Y_%m_%d_%H_%M_%S_%f.png")
        path_string = "./current_imgs/" + prefix + "img_" + f
        # print(path_string)
        cv2.imwrite(path_string, self.raw_cv_image)

    def SLOT_il_recording_button_handler(self):
        self.il_data_counter = 0
        self.il_data_prefix = datetime.now().strftime("./il_data/%Y-%m-%d-%H-%M-%S")
        # print("IL data prefix: " + self.il_data_prefix)

        self.il_recording = not self.il_recording

        if self.il_recording:
            self.il_data_recorder = ImageVectorRecorder(self.il_data_prefix)
            self.il_recording_button.setText("Stop Recording")
        else:
            self.il_data_recorder = None
            self.il_recording_button.setText("Start Recording")

    def SLOT_sign_reading_button_handler(self):
        self.sign_reading = not self.sign_reading
        if self.sign_reading:
            self.sign_reading_button.setText("On")
        else:
            self.sign_reading_button.setText("Off")

    def SLOT_save_images_button_handler(self):
        self.save_current_image()

    def SLOT_reset_state_and_clues(self):
        self.started = False
        self.all_clues_published = False
        self.state_machine = StateMachine()
        self.clue_publisher = ClueTracker()
        self.yoda_threshold_detector = YodaThresholdDetector()
        self.section_2_frame_counter = Counter(SECTION_2_FRAME_COUNT)

    def SLOT_autopilot_button_handler(self):
        self.autopilot = not self.autopilot

        if self.autopilot:
            self.autopilot_button.setText("On")
        else:
            self.autopilot_button.setText("Off")

    def process_clue(self, boxes):
        names = self.yolo_model.names
        data = list(zip((names[id] for id in boxes.cls.tolist()), boxes.xywhn.tolist()))

        data = [t for t in data if t[0] != "sign"]

        if len(data) == 0:
            return

        # Cluster by distance
        coords = np.array([np.array([t[1][0], t[1][1]]) for t in data])

        db = DBSCAN(eps=CLUSTER_EPS, min_samples=2).fit(coords)

        clusters = {}

        for label in db.labels_:
            clusters[label] = []

        for pair, label in zip(data, db.labels_):
            if label == -1:
                continue
            clusters[label].append(pair)

        clusters_with_centroid = []

        for label, items in clusters.items():
            if len(items) == 0:
                continue

            xy = np.array([[item[1][0], item[1][1]] for item in items])
            centroid = xy.mean(axis=0)

            clusters_with_centroid.append(
                (centroid, sorted(items, key=lambda t: t[1][0]))
            )

        # print(clusters_with_centroid)

        sorted_clusters = sorted(clusters_with_centroid, key=lambda t: t[0][1])

        if len(sorted_clusters) == 0:
            return

        clue_type = "".join([t[0] for t in sorted_clusters[0][1]])

        sorted_clusters = sorted_clusters[1:]

        sorted_clusters = sorted(sorted_clusters, key=lambda t: t[0][0])

        # clue = " ".join(
        # ["".join([[t[0] for t in cluster[1]] for cluster in sorted_clusters])]
        # )
        clue = [[t[0] for t in cluster[1]] for cluster in sorted_clusters]
        clue = ["".join(x) for x in clue]
        clue = " ".join(clue)

        # [[t[0] for t in cluster[1]] for cluster in sorted_clusters]

        # print("clue type: " + clue_type)
        # print("clue: " + clue)

        self.latest_clue = (clue_type, clue)

        self.clue_type_label.setText(clue_type)
        self.clue_label.setText(clue)

        self.clue_publisher.addReading(clue_type, clue, 0)

        # print("Clue type: " + clue_type)
        # print("Clue: " + clue)

        # print([[t[0] for t in cluster[1]] for cluster in sorted_clusters])

        # output_string = "".join(t[0] for t in sorted_data)
        # print(output_string)

    def draw_yolo12_detections_bgr(
        self,
        img_bgr,
        conf_thres=0.4,
        iou_thres=0.65,
        recursed=False,
        text_scale=1.4,
        text_line_thickness=2,
    ):
        """
        img_bgr: numpy array in BGR (as from cv2.imread / camera)
        returns: new BGR image with boxes and labels drawn
        """
        # Run inference (Ultralytics accepts numpy BGR directly)
        results = self.yolo_model(
            img_bgr,
            imgsz=640,
            conf=conf_thres,
            iou=iou_thres,
            verbose=False,
        )

        # results is a list; we passed one image → take first result
        r = results[0]

        # Make a copy so we don't scribble over the original
        im_draw = img_bgr.copy()

        # Iterate over detections
        boxes = r.boxes  # Boxes object
        names = self.yolo_model.names  # dict: class_id -> class_name

        if recursed:
            self.process_clue(boxes)

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x0, y0, x1, y1 = box.xyxy[0].tolist()  # [x_min, y_min, x_max, y_max]

            # Convert to int for OpenCV
            x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))

            # If sign is detected and this is top level call, recurse into each sign.
            if cls_id == 36 and not recursed:
                # print("Recursing to read sign!")

                self.annotated_sign_img = self.draw_yolo12_detections_bgr(
                    img_bgr[y0:y1, x0:x1],
                    recursed=True,
                    text_scale=0.4,
                    text_line_thickness=1,
                )

                # print(self.raw_cv_image.shape)
                resized = cv2.resize(self.annotated_sign_img, (400, 261))
                pixmap = self.convert_cv_to_pixmap(resized)
                # print(resized.shape)
                self.sign_image_label.setPixmap(pixmap)

            label = f"{names.get(cls_id, cls_id)} {conf:.2f}"

            # Draw rectangle
            cv2.rectangle(im_draw, (x0, y0), (x1, y1), (0, 255, 0), 2)

            # Draw label background
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_line_thickness
            )
            th = int(th * 1.4)
            cv2.rectangle(im_draw, (x0, y0 - th), (x0 + tw, y0), (0, 255, 0), -1)

            # Put label text (in black on green box)
            cv2.putText(
                im_draw,
                label,
                (x0, y0 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                (0, 0, 0),
                text_line_thickness,
                cv2.LINE_AA,
            )

        return im_draw


if __name__ == "__main__":
    print(cv2.__version__)
    rospy.init_node("robot_view", anonymous=True)
    app = QtWidgets.QApplication(sys.argv)
    robotViewApp = RobotViewApp()
    robotViewApp.show()
    sys.exit(app.exec_())
