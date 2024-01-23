#correct the paths accordingly
from ultralytics import YOLO



model = YOLO('/content/drive/MyDrive/person_dataset/runs/detect/train/weights/best.pt')


model.export(format='onnx')