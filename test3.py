import sys
import cv2
from collections import defaultdict
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from ultralytics import YOLO
from openai import OpenAI
import os

class YOLOApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("smart crack detection")
        self.resize(1280, 1080)
        self.layout = QVBoxLayout()

        self.btn_load = QPushButton("open img")
        self.btn_load.setFixedHeight(50)
        self.btn_load.clicked.connect(self.load_image)
        self.layout.addWidget(self.btn_load)

        self.btn_send = QPushButton("send_result")
        self.btn_send.setFixedHeight(50)
        self.btn_send.clicked.connect(self.send_result)
        self.layout.addWidget(self.btn_send)

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("background-color: black;")
        self.layout.addWidget(self.label)

        self.text_label = QLabel()
        
        font = QFont("Malgun Gothic", 12)
        self.text_label.setFont(font)
        self.label.setFont(font)
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.text_label.setWordWrap(True)
        self.text_label.setStyleSheet("background-color: white; color: black; font-size: 22px;")
        self.text_label.setMinimumHeight(150)
        self.layout.addWidget(self.text_label)

        self.setLayout(self.layout)

        self.model = YOLO("/home/user/ai_hackathon/runs/segment/final2/weights/best.pt")

        self.summery_list = []

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "select img", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        if file_path:
            img = cv2.imread(file_path)
            results = self.model(img)
            res_img = results[0].plot()

            # ----------- 클래스별 개수 & 평균 정확도 -----------
            counts = defaultdict(int)
            conf_sums = defaultdict(float)

            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                counts[cls_id] += 1
                conf_sums[cls_id] += conf

            summary_list = self.summery_list
            summary_list.clear()
            for cls_id, count in counts.items():
                avg_conf = conf_sums[cls_id] / count
                summary_list.append({
                    "class_name": self.model.names[cls_id],
                    "count": count,
                    "avg_confidence": avg_conf
                })

            # print("클래스별 탐지 요약:", summary_list)
            # --------------------------------------------

            # OpenCV BGR → QImage 변환
            h, w, ch = res_img.shape
            bytes_per_line = ch * w
            qimg = QImage(res_img.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)

            pixmap = QPixmap.fromImage(qimg)
            scaled_pixmap = pixmap.scaled(
                self.label.width(), self.label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.label.setPixmap(scaled_pixmap)

    
    def send_result(self):
        api_key = os.getenv("OPENAI_API_KEY")
        summary_list = self.summery_list
        
        # OpenAI 클라이언트 생성
        client = OpenAI(api_key=api_key)

        # GPT 모델에 요청
        response = client.chat.completions.create(
            model="gpt-4.1-mini",  # 원하는 모델 지정
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
                {"role": "user", "content": str(summary_list)}
            ],
            max_tokens=200,
            temperature=0.1  
        )

        # 출력
        self.text_label.setText(response.choices[0].message.content)
        


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOApp()
    window.show()
    sys.exit(app.exec())