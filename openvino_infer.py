#/home/sanya/Downloads/best_openvino_model-20240117T081318Z-001/best_openvino_model/best.xml
#correct the paths accordingly
import cv2
from openvino.inference_engine import IECore
import time


model_xml = "/home/sanya/Downloads/best_openvino_model-20240117T081318Z-001/best_openvino_model/best.xml"
model_bin = "/home/sanya/Downloads/best_openvino_model-20240117T081318Z-001/best_openvino_model/best.bin"


ie = IECore()
net = ie.read_network(model=model_xml, weights=model_bin)


input_blob = next(iter(net.input_info))
output_blob = next(iter(net.outputs))

# Set up the network
exec_net = ie.load_network(network=net, device_name="CPU")

cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error opening camera"

start_time = time.time()
frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Error reading camera frame.")
        break

    input_frame = cv2.resize(frame, (net.input_info[input_blob].input_data.shape[3], net.input_info[input_blob].input_data.shape[2]))
    input_frame = input_frame.transpose((2, 0, 1))
    input_frame = input_frame.reshape(1, *input_frame.shape)


    infer_start = time.time()
    result = exec_net.infer(inputs={input_blob: input_frame})
    infer_time = time.time() - infer_start


    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow("OpenVINO Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
