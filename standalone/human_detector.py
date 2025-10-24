from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture(0)
model = YOLO('yolov8n.pt')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    for box in results[0].boxes:
        if int(box.cls[0]) == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('YOLO Detected', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()