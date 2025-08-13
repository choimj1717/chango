import sys
import cv2
import os
from collections import defaultdict
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import Qt, QTimer
from ultralytics import YOLO
from openai import OpenAI

# Wayland 오류 방지
os.environ["QT_QPA_PLATFORM"] = "xcb"

class YOLOApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Crack Detection")
        self.resize(1280, 1080)
        self.layout = QVBoxLayout()

        # Load Image Button
        self.btn_load = QPushButton("Open Image")
        self.btn_load.setFixedHeight(50)
        self.btn_load.clicked.connect(self.load_image)
        self.layout.addWidget(self.btn_load)

        # Camera Start Button
        self.btn_camera = QPushButton("Start Camera")
        self.btn_camera.setFixedHeight(50)
        self.btn_camera.clicked.connect(self.start_camera)
        self.layout.addWidget(self.btn_camera)

        # Label to display image
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("background-color: black;")
        self.layout.addWidget(self.label)

        # Text label for results
        self.text_label = QLabel()
        font = QFont("Malgun Gothic", 12)
        self.text_label.setFont(font)
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.text_label.setWordWrap(True)
        self.text_label.setStyleSheet("background-color: white; color: black; font-size: 22px;")
        self.text_label.setMinimumHeight(150)
        self.layout.addWidget(self.text_label)

        self.setLayout(self.layout)

        # Load YOLO model (v8)
        self.model = YOLO("/home/pi/Downloads/test_crack/chango/seg_model_1.pt")

        # Store results
        self.summary_list = []

        # Camera variables
        self.capture = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # Store last captured frame
        self.last_captured_frame = None

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        if file_path:
            img = cv2.imread(file_path)
            self.process_image(img)

    def process_image(self, img):
        # YOLOv8 추론 (속도 최적화 위해 imgsz=640)
        results = self.model(img, imgsz=640)
        res_img = results[0].plot()

        counts = defaultdict(int)
        conf_sums = defaultdict(float)
        detected = False

        for box in results[0].boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            counts[cls_id] += 1
            conf_sums[cls_id] += conf
            if conf > 0.5:
                detected = True

        self.summary_list.clear()
        for cls_id, count in counts.items():
            avg_conf = conf_sums[cls_id] / count
            self.summary_list.append({
                "class_name": self.model.names[cls_id],
                "count": count,
                "avg_confidence": avg_conf
            })

        # 감지 시 자동 저장 + GPT 전송
        if detected:
            self.last_captured_frame = img.copy()
            save_path = "/tmp/captured_frame.jpg"
            cv2.imwrite(save_path, self.last_captured_frame)
            self.text_label.setText(f"Object detected! Saved to {save_path}")
            if self.capture:
                self.capture.release()
                self.timer.stop()
            self.send_result()

        # PyQt6에 표시
        h, w, ch = res_img.shape
        qimg = QImage(res_img.data, w, h, ch * w, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.label.width(), self.label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.label.setPixmap(pixmap)

    def start_camera(self):
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            self.text_label.setText("Failed to open camera.")
            return
        self.timer.start(30)
        self.text_label.setText("Camera started. Detecting objects...")

    def update_frame(self):
        if not self.capture or not self.capture.isOpened():
            return
        ret, frame = self.capture.read()
        if ret:
            self.process_image(frame)

    def send_result(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.text_label.setText("OPENAI_API_KEY not set!")
            return
        if self.last_captured_frame is None:
            self.text_label.setText("No captured frame to send.")
            return

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": """당신은 건축물관리사입니다.
                 yolo 모델의 추론 결과를 바탕으로 건축물의 상태를 평가하고, 필요한 조치를 제안합니다.
                 yolo 모델의 클래스는 crack, spalling 2가지입니다.
                 yolo 모델의 추론 결과의 예시이자 입력 형식의 예시는 다음과 같습니다.
                 
                 {'class_name': 'cracks', 'count': 3, 'avg_confidence': 0.8542},
                 {'class_name': 'spailling', 'count': 1, 'avg_confidence': 0.9215}
                 
                 crack은 균열을 의미하며, spalling은 건축물의 박리 현상을 의미합니다.
                 만약 spalling이 탐지되었다면 높은 위험도로 판답합니다.
                 출력 형식은 다음과 같습니다.

                    건축물의 위험도 : [위험도%]
                    판단 이유 : [이유]
                    조치 사항 : [조치사항]
                 
                 
                 """},
                {"role": "user", "content": str(self.summary_list)}
            ],
            max_tokens=200,
            temperature=0.1
        )

        self.text_label.setText(response.choices[0].message.content)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOApp()
    window.show()
    sys.exit(app.exec())
