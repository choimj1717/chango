import sys
import cv2
import os
from collections import defaultdict
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import Qt, QTimer
from ultralytics import YOLO
from openai import OpenAI
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

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

        # __init__ 내부에 추가
        self.detected_images = []  # 감지된 이미지 경로 저장
        self.btn_send = QPushButton("Send Email")
        self.btn_send.setFixedHeight(50)
        self.btn_send.clicked.connect(self.send_all_results)
        self.layout.addWidget(self.btn_send)


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
            if conf > 0.6:
                detected = True

        self.summary_list.clear()
        for cls_id, count in counts.items():
            avg_conf = conf_sums[cls_id] / count
            self.summary_list.append({
                "class_name": self.model.names[cls_id],
                "count": count,
                "avg_confidence": avg_conf
            })

        # 감지 시 자동 저장 (이메일 전송은 안 함)
        if detected:
            self.last_captured_frame = img.copy()
            save_path = f"/tmp/captured_frame_{len(self.detected_images)+1}.jpg"
            cv2.imwrite(save_path, self.last_captured_frame)
            self.detected_images.append(save_path)
            self.text_label.setText(f"Detected! Saved {save_path}")

        # 화면 표시
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

    def send_all_results(self):
        """모아둔 이미지를 한 번에 GPT 분석 + 이메일 전송"""
        if not self.detected_images:
            self.text_label.setText("No images to send.")
            return

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.text_label.setText("OPENAI_API_KEY not set!")
            return

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": """
                당신은 건축물관리사입니다.
                yolo 모델의 추론 결과를 바탕으로 건축물의 상태를 평가하고, 필요한 조치를 제안합니다.
                클래스: crack(균열), spalling(박리). spalling 탐지 시 높은 위험도.
                출력 예시:
                건축물의 위험도 : [위험도%]
                판단 이유 : [이유]
                조치 사항 : [조치사항]
                """},
                {"role": "user", "content": str(self.summary_list)}
            ],
            max_tokens=300,
            temperature=0.1
        )

        gpt_result = response.choices[0].message.content
        self.text_label.setText(gpt_result)

        # 이메일 전송
        self.send_email(
            subject="YOLO Crack Detection Report - Batch",
            body=gpt_result,
            attachment_paths=self.detected_images
        )

        # 전송 후 초기화
        self.detected_images.clear()

        
    def send_email(self, subject, body, attachment_paths):
        sender_email = "choiyou1717@gmail.com"
        receiver_email = "choimj3909@gmail.com"
        password = "kfqm olmt gteq fqmc"

        # kfqm olmt gteq fqmc
        
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        # 여러 이미지 첨부
        for path in attachment_paths:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(path)}")
                msg.attach(part)

        try:
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.starttls()
            server.login(sender_email, password)
            server.send_message(msg)
            server.quit()
            print("Email sent successfully!")
        except Exception as e:
            print("Email sending failed:", e)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOApp()
    window.show()
    sys.exit(app.exec())
