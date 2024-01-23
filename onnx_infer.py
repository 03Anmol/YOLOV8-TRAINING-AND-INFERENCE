import cv2
import time
import onnxruntime
import numpy as np

# Load the ONNX model
onnx_model_path = "/home/sanya/Downloads/best.onnx"
ort_session = onnxruntime.InferenceSession(onnx_model_path)
input_names = [input.name for input in ort_session.get_inputs()]
print("Model input names:", input_names)

cap = cv2.VideoCapture(0)  
assert cap.isOpened(), "Error opening camera"
start_time = time.time()
frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Error reading camera frame.")
        break
    input_data = cv2.resize(frame, (640, 640)) 
    input_data = input_data.transpose(2, 0, 1)  
    input_data = np.expand_dims(input_data, axis=0)  
    input_data = input_data.astype(np.float32) / 255.0  

    ort_inputs = {input_names[0]: input_data} 
    ort_outputs = ort_session.run(None, ort_inputs)
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 ONNX Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
