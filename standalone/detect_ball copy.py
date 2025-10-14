import cv2
import numpy as np
import sys

# ===== 設定 =====
mode = "video"  # "video", "camera"
ball_color = "white"  # "white", "orange", "both"
resize_width = 640
resize_height = 480
video_path = "standalone/video/ping.mp4"
output_path = "standalone/video/output_detected.mp4"

# 検出パラメータ
min_radius = 5
max_radius = 50
min_area = 50
max_area = 3000
circularity_threshold = 0.6  # 円形度（0-1、1が完全な円）
jitter_threshold = 100  # 急激な移動を除外（ピクセル）
lost_frame_threshold = 15  # この回数まで予測で追跡継続

# ===== 初期化 =====
if mode == "camera":
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Cannot open video source")
    sys.exit(1)

# FPS取得と出力設定
fps = cap.get(cv2.CAP_PROP_FPS) if mode == "video" else 30.0
print(f"FPS: {fps}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (resize_width, resize_height))

# 色範囲の設定
color_ranges = []
if ball_color in ["white", "both"]:
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    color_ranges.append((lower_white, upper_white, "White"))

if ball_color in ["orange", "both"]:
    lower_orange = np.array([5, 150, 150])
    upper_orange = np.array([20, 255, 255])
    color_ranges.append((lower_orange, upper_orange, "Orange"))

# カルマンフィルタの初期化
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], 
                                     [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

# 追跡状態
trajectory = []
max_traj_length = 30
prev_center = None
prev_radius = None
lost_frames = 0
detected_color = None

# ===== ヘルパー関数 =====
def calculate_circularity(contour):
    """輪郭の円形度を計算"""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0
    circularity = 4 * np.pi * area / (perimeter ** 2)
    return circularity

def is_motion_valid(center):
    """動きの妥当性をチェック"""
    if prev_center is None:
        return True
    
    distance = np.sqrt((center[0] - prev_center[0])**2 + 
                      (center[1] - prev_center[1])**2)
    return distance < jitter_threshold

def is_size_valid(radius):
    """サイズの妥当性をチェック"""
    if prev_radius is None:
        return True
    
    ratio = radius / prev_radius if prev_radius > 0 else 1
    return 0.5 < ratio < 2.0

def find_best_ball(contours, prediction=None):
    """最も卓球ボールらしい輪郭を見つける"""
    best_candidate = None
    best_score = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # 面積フィルタ
        if area < min_area or area > max_area:
            continue
        
        # 円形度チェック
        circularity = calculate_circularity(cnt)
        if circularity < circularity_threshold:
            continue
        
        # 外接円を取得
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        
        # 半径フィルタ
        if radius < min_radius or radius > max_radius:
            continue
        
        # 動きの妥当性チェック
        if not is_motion_valid(center):
            continue
        
        # サイズの妥当性チェック
        if not is_size_valid(radius):
            continue
        
        # スコア計算
        score = circularity
        
        # 予測位置との距離を考慮
        if prediction is not None:
            pred_x, pred_y = int(prediction[0]), int(prediction[1])
            dist = np.sqrt((center[0] - pred_x)**2 + (center[1] - pred_y)**2)
            dist_score = 1.0 / (1.0 + dist / 50.0)
            score = circularity * 0.6 + dist_score * 0.4
        
        if score > best_score:
            best_score = score
            best_candidate = (center, radius, score, circularity)
    
    return best_candidate

# ===== メインループ =====
frame_count = 0
print("Processing... Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or camera error")
        break
    
    frame_count += 1
    frame = cv2.resize(frame, (resize_width, resize_height))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 色範囲でマスク作成
    mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper, color_name in color_ranges:
        mask = cv2.inRange(hsv, lower, upper)
        mask_total = cv2.bitwise_or(mask_total, mask)
    
    # ノイズ除去
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 輪郭検出
    contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)
    
    # カルマンフィルタで予測
    prediction = kalman.predict()
    
    # 最適な候補を探す
    ball_candidate = find_best_ball(contours, prediction)
    
    if ball_candidate is not None:
        center, radius, score, circularity = ball_candidate
        
        # カルマンフィルタを更新
        measured = np.array([[np.float32(center[0])], 
                            [np.float32(center[1])]])
        kalman.correct(measured)
        
        # 軌跡に追加
        trajectory.append(center)
        prev_center = center
        prev_radius = radius
        lost_frames = 0
        
        # 描画
        cv2.circle(frame, center, radius, (0, 255, 0), 2)
        cv2.circle(frame, center, 3, (0, 0, 255), -1)
        
        # 情報表示
        info_text = f"Ball R:{radius} C:{circularity:.2f}"
        cv2.putText(frame, info_text, (center[0] - 50, center[1] - radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    else:
        # 検出できない場合
        lost_frames += 1
        
        if lost_frames < lost_frame_threshold:
            # 予測位置を使用
            pred_x, pred_y = int(prediction[0]), int(prediction[1])
            center = (pred_x, pred_y)
            trajectory.append(center)
            
            # 予測位置を薄く表示
            cv2.circle(frame, center, 10, (0, 165, 255), 2)
            cv2.putText(frame, "Predicted", (center[0] - 30, center[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        else:
            # 長時間見失った場合はリセット
            prev_center = None
            prev_radius = None
    
    # 古い軌跡を削除
    if len(trajectory) > max_traj_length:
        trajectory = trajectory[-max_traj_length:]
    
    # 軌跡を描画
    for i in range(1, len(trajectory)):
        if trajectory[i-1] is None or trajectory[i] is None:
            continue
        thickness = int(np.sqrt(max_traj_length / float(i + 1)) * 2)
        cv2.line(frame, trajectory[i-1], trajectory[i], (0, 255, 255), thickness)
    
    # ステータス表示
    status_color = (0, 255, 0) if lost_frames == 0 else (0, 165, 255)
    status_text = "TRACKING" if lost_frames == 0 else f"LOST {lost_frames}"
    cv2.putText(frame, status_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    cv2.putText(frame, f"Frame: {frame_count}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.putText(frame, f"Candidates: {len(contours)}", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 保存と表示
    out.write(frame)
    cv2.imshow('Tracking', frame)
    cv2.imshow('Mask', mask_total)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quit by user")
        break

# クリーンアップ
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Output saved to: {output_path}")
print(f"Total frames processed: {frame_count}")