import cv2
from ultralytics import YOLO
from deepface import DeepFace
import time

# モデルの読み込み
yolo_model = YOLO('yolov8n.pt')  # 軽量モデル

# Webカメラの初期化
cap = cv2.VideoCapture(0)  # 0はデフォルトのWebカメラ
if not cap.isOpened():
    raise RuntimeError("Webカメラを開けませんでした。")

# 高速化のための設定
FRAME_SKIP = 2  # 年齢推定を2フレームごとに実行
RESIZE_SCALE = 0.5  # 画像サイズを半分に縮小
frame_count = 0

while True:
    start_time = time.time()
    
    # フレームの取得
    ret, frame = cap.read()
    if not ret:
        print("フレームの取得に失敗しました。")
        break
    
    # 画像サイズの縮小
    height, width = frame.shape[:2]
    frame_resized = cv2.resize(frame, (int(width * RESIZE_SCALE), int(height * RESIZE_SCALE)))
    
    # 人物検出
    results = yolo_model(frame_resized, classes=[0], verbose=False)  # クラス0は人物
    
    for result in results:
        boxes = result.boxes.xyxy  # バウンディングボックスの座標
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            # 縮小した座標を元画像にスケールバック
            x1, y1, x2, y2 = [int(coord / RESIZE_SCALE) for coord in [x1, y1, x2, y2]]
            
            # バウンディングボックスを描画
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 年齢推定（FRAME_SKIPごとに実行）
            if frame_count % FRAME_SKIP == 0:
                try:
                    person_crop = frame[y1:y2, x1:x2]
                    if person_crop.size > 0:  # 空の画像を回避
                        analysis = DeepFace.analyze(person_crop, actions=['age'], enforce_detection=True, 
                                                 detector_backend='opencv', silent=True)
                        age = analysis[0]['age']
                        cv2.putText(frame, f'Age: {age}', (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                except Exception as e:
                    cv2.putText(frame, 'No face detected', (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # FPSの計算と表示
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # フレームを表示
    cv2.imshow('Real-time Detection', frame)
    
    # フレームカウントを更新
    frame_count += 1
    
    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースの解放
cap.release()
cv2.destroyAllWindows()
print("プログラムを終了しました。")