import cv2
from ultralytics import YOLO
from deepface import DeepFace

# モデルの読み込み
yolo_model = YOLO('yolov8n.pt')  # YOLOv8の軽量モデル

# 画像の読み込み
image_path = 'pic3.jpg'
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"画像 {image_path} が見つかりません。")

# 人物検出
results = yolo_model(image, classes=[0])  # クラス0は人物
output_image = image.copy()

for result in results:
    boxes = result.boxes.xyxy  # バウンディングボックスの座標
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        # 人物領域を切り出し
        person_crop = image[y1:y2, x1:x2]
        
        try:
            # 顔検出と年齢推定
            analysis = DeepFace.analyze(person_crop, actions=['age'], enforce_detection=True)
            age = analysis[0]['age']
            
            # バウンディングボックスと年齢を描画
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output_image, f'Age: {age}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except Exception as e:
            # 顔検出に失敗した場合
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(output_image, 'No face detected', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

# 結果を保存
cv2.imwrite('output.jpg', output_image)
print("結果を output.jpg に保存しました。")