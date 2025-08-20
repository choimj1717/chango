# cam_yolo.py
from ultralytics import YOLO

# 모델 로드
model = YOLO("seg_model_1.pt")

# 웹캠에서 실시간 추론
model.predict(
    source=1,      # 0은 기본 웹캠
    conf=0.5,      # confidence threshold (0~1)
    show=True,     # 추론 결과 창에 표시
    save=False     # 저장하지 않음, True로 바꾸면 runs/predict 폴더에 저장
)
