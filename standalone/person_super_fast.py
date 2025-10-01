import cv2
from ultralytics import YOLO
from deepface import DeepFace
import mediapipe as mp
import time
import numpy as np

# モデルの読み込み
yolo_model = YOLO('yolov8n.pt')  # 軽量モデル
mp_face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.3)

# Webカメラの初期化
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # 解像度をさらに下げる
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
if not cap.isOpened():
    raise RuntimeError("Webカメラを開けませんでした。")

# 高速化のための設定
FRAME_SKIP = 10  # 年齢推定を10フレームごとに実行
RESIZE_SCALE = 0.3  # 画像サイズを30%に縮小
frame_count = 0
age_cache = {}  # トラッキングIDごとの年齢キャッシュ
track_id = 0  # 簡易トラッキングID

def get_box_center(x1, y1, x2, y2):
    """バウンディングボックスの中心座標を計算"""
    return (x1 + x2) // 2, (y1 + y2) // 2

def assign_track_id(center, existing_centers, threshold=50):
    """簡易トラッキング：中心座標が近い場合、同じIDを割り当て"""
    for cid, ccenter in existing_centers.items():
        dist = np.sqrt((center[0] - ccenter[0])**2 + (center[1] - ccenter[1])**2)
        if dist < threshold:
            return cid
    return None

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
    
    current_centers = {}  # 現在のフレームの中心座標
    for result in results:
        boxes = result.boxes.xyxy  # バウンディングボックスの座標
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            # 縮小した座標を元画像にスケールバック
            x1, y1, x2, y2 = [int(coord / RESIZE_SCALE) for coord in [x1, y1, x2, y2]]
            
            # バウンディングボックスを描画
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # トラッキングIDの割り当て
            box_center = get_box_center(x1, y1, x2, y2)
            tid = assign_track_id(box_center, current_centers)
            if tid is None:
                tid = track_id
                current_centers[tid] = box_center
                track_id += 1
            
            # 年齢推定（FRAME_SKIPごと、またはキャッシュにない場合）
            if frame_count % FRAME_SKIP == 0 or tid not in age_cache:
                # MediaPipeで顔検出
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size > 0:
                    person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                    results_face = mp_face_detection.process(person_crop_rgb)
                    
                    if results_face.detections:
                        try:
                            # DeepFaceで年齢推定
                            analysis = DeepFace.analyze(person_crop, actions=['age'], enforce_detection=True, 
                                                     detector_backend='opencv', silent=True)
                            age = analysis[0]['age']
                            age_cache[tid] = age  # キャッシュに保存
                            cv2.putText(frame, f'ID: {tid} Age: {age}', (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        except Exception:
                            cv2.putText(frame, f'ID: {tid} No face', (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, f'ID: {tid} No face', (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # キャッシュから年齢を取得
                if tid in age_cache:
                    age = age_cache[tid]
                    cv2.putText(frame, f'ID: {tid} Age: {age}', (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # FPSの計算と表示
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # フレームを表示
    cv2.imshow('Real-time Detection', frame)
    
    # フレームカウントを更新
    frame_count += 1
    
    # キャッシュのクリア（メモリ節約のため、100フレーム後にリセット）
    if frame_count % 100 == 0:
        age_cache.clear()
    
    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースの解放
cap.release()
cv2.destroyAllWindows()
print("プログラムを終了しました。")