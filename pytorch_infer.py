from ultralytics import YOLO
import cv2
import time
model = YOLO("/home/sanya/Downloads/best.pt")

cap = cv2.VideoCapture(0) 
assert cap.isOpened(), "Error opening camera"

start_time = time.time()
frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Error reading camera frame.")
        break
    results = model.predict(frame)

    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLOv5 Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
