import cv2
import numpy as np

mode = "video" # "video", "camera"
ball_color = "white"  # "white", "orange", "both"
resize_width = 640
resize_height = 480
video_path = "standalone/video/ping.mp4"
# カメラまたは動画ファイルの選択
if mode == "camera":
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(video_path)
# 保存する動画ファイル名と設定
output_path = "standalone/video/output_detected.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
if mode == "video":
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Input video FPS: {fps}")
out = cv2.VideoWriter(output_path, fourcc, fps, (resize_width, resize_height))

# 色範囲の設定
color_ranges = []
if ball_color in ["white", "both"]:
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    color_ranges.append((lower_white, upper_white))
if ball_color in ["orange", "both"]:
    lower_orange = np.array([5, 150, 150])
    upper_orange = np.array([20, 255, 255])
    color_ranges.append((lower_orange, upper_orange))

prev_mask = None

# カルマンフィルタの初期化
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

trajectory = []
max_traj_length = 30  # 軌跡の最大フレーム数（例: 30フレーム分だけ描画）
# ぶれ判定用のしきい値（ピクセル単位、調整可）
jitter_threshold = 20

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (resize_width, resize_height))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lower, upper in color_ranges:
        mask = cv2.inRange(hsv, lower, upper)
        mask_total = cv2.bitwise_or(mask_total, mask)

    # mask_total = cv2.medianBlur(mask_total, 5)
    mask_total = cv2.erode(mask_total, None, iterations=1)
    mask_total = cv2.dilate(mask_total, None, iterations=1)

    if prev_mask is not None:
        diff = cv2.absdiff(mask_total, prev_mask)
        _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    else:
        diff = mask_total.copy()

    prev_mask = mask_total.copy()

    contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    measured = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            if 10 < radius < 50:
                measured = np.array([[np.float32(x)], [np.float32(y)]])
                cv2.circle(frame, center, radius, (0, 255, 0), 2)
                cv2.putText(frame, "Ball", (center[0]-10, center[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                break  # 最初に見つかったボールのみ追跡

    # カルマンフィルタによる予測と補正
    prediction = kalman.predict()
    pred_x, pred_y = int(prediction[0]), int(prediction[1])

    if measured is not None:
        kalman.correct(measured)
        trajectory.append((int(measured[0][0]), int(measured[1][0])))
    else:
        trajectory.append((pred_x, pred_y))

    # 古い軌跡を削除
    if len(trajectory) > max_traj_length:
        trajectory = trajectory[-max_traj_length:]

    # 軌跡を描画
    for i in range(1, len(trajectory), 1):
        cv2.line(frame, trajectory[i-1], trajectory[i], (0,0,255), 2)

    out.write(frame)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', diff)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()