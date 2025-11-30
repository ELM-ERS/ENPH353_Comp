import roslib
import rospy
import sys
import numpy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from datetime import datetime

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
import torch
import numpy as np
from sklearn.cluster import DBSCAN


class RobotViewApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(RobotViewApp, self).__init__()
        loadUi("./robot_view.ui", self)

        # self.yolo_model = YOLO("./main_sign_model.pt")
        self.yolo_model = YOLO("./main_sign_model.torchscript")

        # self.yolo_model = DetectionModel("yolo12m.yaml")
        # ckpt = torch.load("./main_sign_model.pt", map_location="cuda")
        # self.yolo_model.load_state_dict(ckpt["model"])

        self.bridge = CvBridge()

        self.latest_clue = ("", "")

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

            self.annotated_image = self.draw_yolo12_detections_bgr(self.raw_cv_image)
            if self.recording:
                self.save_current_image(prefix="recording/")
        except CvBridgeError as e:
            print(e)

        # print(self.raw_cv_image.shape)
        resized = cv2.resize(self.annotated_image, (400, 400))
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

    def process_clue(self, boxes):
        names = self.yolo_model.names
        data = list(zip((names[id] for id in boxes.cls.tolist()), boxes.xywhn.tolist()))

        data = [t for t in data if t[0] != "sign"]

        if len(data) == 0:
            return

        # Cluster by distance
        coords = np.array([np.array([t[1][0], t[1][1]]) for t in data])

        db = DBSCAN(eps=0.14, min_samples=2).fit(coords)

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

        print("clue type: " + clue_type)
        print("clue: " + clue)

        self.latest_clue = (clue_type, clue)

        self.clue_type_label.setText(clue_type)
        self.clue_label.setText(clue)

        # print("Clue type: " + clue_type)
        # print("Clue: " + clue)

        # print([[t[0] for t in cluster[1]] for cluster in sorted_clusters])

        # output_string = "".join(t[0] for t in sorted_data)
        # print(output_string)

    def draw_yolo12_detections_bgr(
        self,
        img_bgr,
        conf_thres=0.25,
        iou_thres=0.45,
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
                print("Recursing to read sign!")

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
