# main_tflite.py
import cv2
import numpy as np
import time
from imutils.video import VideoStream
from config import tflite # Universal AI Engine

# --- CONFIGURATION ---
PROTO_PATH = "face_detector/deploy.prototxt"
MODEL_PATH = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
EYE_MODEL_PATH = "models/pretrained/eye_yolo.tflite" # Path to your YOLO model
CONFIDENCE_THRESHOLD = 0.5

# 1. LOAD MODELS
print("[INFO] Loading Face Detector...")
face_net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)

print("[INFO] Loading Eye AI Model...")
# Technical term: Interpreter Initialization
interpreter = tflite.Interpreter(model_path=EYE_MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details for the AI
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 2. START VIDEO STREAM
vs = VideoStream(src=0).start()
time.sleep(2.0)
prev_time = 0

while True:
    frame = vs.read()
    if frame is None: break
    (h, w) = frame.shape[:2]

    # STAGE 1: FACE DETECTION
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > CONFIDENCE_THRESHOLD:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # STAGE 2: EYE ROI EXTRACTION
            face_height = endY - startY
            eye_roi = frame[startY : startY + (face_height // 2), startX : endX]

            if eye_roi.size > 0:
                # STAGE 3: AI INFERENCE (Is the driver drowsy?)
                # 3a. Preprocessing (Format the image for YOLO)
                # YOLOv8 expects 320x320 RGB images
                img_ai = cv2.resize(eye_roi, (320, 320))
                img_ai = cv2.cvtColor(img_ai, cv2.COLOR_BGR2RGB)
                img_ai = np.expand_dims(img_ai, axis=0).astype(np.float32) / 255.0

                # 3b. Run Inference
                interpreter.set_tensor(input_details[0]['index'], img_ai)
                interpreter.invoke()
                
                # 3c. Get Results (Post-processing)
                output_data = interpreter.get_tensor(output_details[0]['index'])
                # Simplified logic: we check the highest probability class
                # Note: Class 0 or 1 depending on your model labels
                prediction = np.argmax(output_data[0]) 
                
                state = "CLOSED" if prediction == 0 else "OPEN"
                color = (0, 0, 255) if state == "CLOSED" else (0, 255, 0)

                # VISUAL FEEDBACK
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.putText(frame, f"Eyes: {state}", (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 4. FPS & DISPLAY
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Driver Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

vs.stop()
cv2.destroyAllWindows()