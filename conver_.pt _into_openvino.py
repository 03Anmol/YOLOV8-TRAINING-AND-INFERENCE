from ultralytics import YOLO

#correct the paths accordingly
model = YOLO('/content/drive/MyDrive/person_dataset/runs/detect/train/weights/best.pt')

model.export(format='openvino')

ov_model = YOLO('/content/drive/MyDrive/person_dataset/runs/detect/train/weights/yolov8n_openvino_model')