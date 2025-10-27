from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture(0)
model = YOLO('yolov8n-pose.pt')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = results[0].plot()

    cv2.imshow('YOLO Detected', annotated)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()