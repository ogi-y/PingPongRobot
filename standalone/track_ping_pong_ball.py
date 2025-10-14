import cv2
import numpy as np
from pathlib import Path

# 設定
VIDEO_PATH = "./standalone/video/ping.mp4"  # 動画ファイルのパス（Noneでカメラ使用）
OUTPUT_PATH = "./standalone/video/pong.mp4"  # 出力動画のパス（Noneで保存しない）
BALL_COLOR = "both"  # "white", "orange", "both"

# ボール検出パラメータ
MIN_RADIUS = 5  # 最小半径（ピクセル）
MAX_RADIUS = 50  # 最大半径（ピクセル）
MIN_CIRCULARITY = 0.7  # 最小円形度（0-1）

# ロボット位置設定（画面下部からのピクセル数、またはパーセンテージ）
ROBOT_Y_POSITION = 0.9  # 画面下部から90%の位置（0.0-1.0）または絶対ピクセル値
TRAJECTORY_HISTORY = 10  # 軌跡履歴のフレーム数

# イベント検出パラメータ
ACCELERATION_THRESHOLD = 30.0  # 加速度の閾値（px/frame^2）打球検出用（高めに設定）
MIN_SPEED_FOR_HIT = 5.0  # 打球検出の最小速度（px/frame）低速時の誤検出を防ぐ
HIT_COOLDOWN_FRAMES = 10  # 打球イベント間の最小フレーム数

# 白ボールのHSV範囲
WHITE_LOWER = np.array([0, 0, 200])
WHITE_UPPER = np.array([180, 30, 255])

# オレンジボールのHSV範囲
ORANGE_LOWER = np.array([5, 100, 100])
ORANGE_UPPER = np.array([20, 255, 255])


class BallTracker:
    """卓球ボールのトラッカー"""
    
    def __init__(self, color_mode="both", max_history=10):
        """
        Args:
            color_mode: "white", "orange", "both"のいずれか
            max_history: 保持する軌跡履歴の最大フレーム数
        """
        self.color_mode = color_mode
        self.ball_trajectories = {}  # ボールごとの軌跡 {ball_id: [(x, y, frame), ...]}
        self.max_history = max_history
        self.next_ball_id = 0
        self.ball_events = {}  # ボールごとのイベント履歴 {ball_id: [events...]}
        
    def update_trajectories(self, balls, frame_number):
        """
        ボールの軌跡を更新
        
        Args:
            balls: 検出されたボールのリスト [(x, y, radius, color), ...]
            frame_number: 現在のフレーム番号
        """
        # 新しい検出に対して最も近い既存の軌跡を探す（簡易マッチング）
        matched_ids = set()
        new_trajectories = {}
        
        for x, y, radius, color in balls:
            # 最も近い既存の軌跡を探す
            min_dist = float('inf')
            best_id = None
            
            for ball_id, trajectory in self.ball_trajectories.items():
                if ball_id in matched_ids:
                    continue
                if len(trajectory) > 0:
                    last_x, last_y, last_frame, last_radius, last_color = trajectory[-1]
                    dist = np.sqrt((x - last_x)**2 + (y - last_y)**2)
                    if dist < min_dist and dist < 100:  # 最大移動距離の閾値
                        min_dist = dist
                        best_id = ball_id
            
            # マッチした軌跡に追加、または新規作成
            if best_id is not None:
                matched_ids.add(best_id)
                trajectory = self.ball_trajectories[best_id].copy()
                
                # 加速度検出（打球検出）
                self._detect_acceleration(best_id, trajectory, x, y, frame_number)
                
                trajectory.append((x, y, frame_number, radius, color))
                # 履歴を制限
                if len(trajectory) > self.max_history:
                    trajectory = trajectory[-self.max_history:]
                new_trajectories[best_id] = trajectory
                
                # バウンド検出
                self._detect_bounce(best_id, trajectory, frame_number)
            else:
                # 新しいボール
                new_id = self.next_ball_id
                self.next_ball_id += 1
                new_trajectories[new_id] = [(x, y, frame_number, radius, color)]
                self.ball_events[new_id] = []  # イベント履歴を初期化
        
        self.ball_trajectories = new_trajectories
    
    def _detect_acceleration(self, ball_id, trajectory, new_x, new_y, frame_number):
        """
        加速度を検出して打球イベントを判定
        重力による変化は無視し、打球による急激な速度変化のみを検出
        
        Args:
            ball_id: ボールID
            trajectory: 現在の軌跡
            new_x, new_y: 新しい位置
            frame_number: 現在のフレーム番号
        """
        # 最低3点必要（2つの速度を計算するため）
        if len(trajectory) < 2:
            return
        
        # 最新の2点から現在の速度を計算
        x1, y1, frame1, _, _ = trajectory[-1]
        x2, y2, frame2, _, _ = trajectory[-2]
        
        dt1 = frame_number - frame1
        dt2 = frame1 - frame2
        
        if dt1 == 0 or dt2 == 0:
            return
        
        # X方向とY方向の速度を個別に計算
        current_vx = (new_x - x1) / dt1
        current_vy = (new_y - y1) / dt1
        
        prev_vx = (x1 - x2) / dt2
        prev_vy = (y1 - y2) / dt2
        
        # X方向の加速度（重力の影響を受けない）
        ax = abs(current_vx - prev_vx) / dt1
        
        # Y方向の加速度変化（重力を考慮）
        # 重力による加速度は一定なので、加速度の「変化」を見る
        # または、X方向の加速度を主に使用
        
        # 速度の大きさ
        current_speed = np.sqrt(current_vx**2 + current_vy**2)
        prev_speed = np.sqrt(prev_vx**2 + prev_vy**2)
        
        # 打球検出条件：
        # 1. X方向の加速度が閾値を超える（重力の影響なし）
        # 2. かつ、ボールが一定以上の速度を持っている（誤検出防止）
        speed_change = abs(current_speed - prev_speed)
        acceleration_rate = speed_change / dt1
        
        # より厳密な条件で判定
        # - X方向の加速度が閾値を超える
        # - かつ、現在の速度が最小値以上
        # - かつ、速度変化率も閾値を超える
        if (ax > ACCELERATION_THRESHOLD and current_speed > MIN_SPEED_FOR_HIT) or \
           (acceleration_rate > ACCELERATION_THRESHOLD and current_speed > MIN_SPEED_FOR_HIT * 1.5):
            # イベント履歴を初期化（必要に応じて）
            if ball_id not in self.ball_events:
                self.ball_events[ball_id] = []
            
            # 最近のイベントと重複しないようにチェック（クールダウン期間を設定）
            recent_hit = False
            for event in reversed(self.ball_events[ball_id]):
                if event['type'] == 'hit' and (frame_number - event['frame']) < HIT_COOLDOWN_FRAMES:
                    recent_hit = True
                    break
            
            if not recent_hit:
                event = {
                    'type': 'hit',
                    'frame': frame_number,
                    'position': (new_x, new_y),
                    'acceleration_x': ax,
                    'speed_change': acceleration_rate,
                    'prev_speed': prev_speed,
                    'current_speed': current_speed
                }
                self.ball_events[ball_id].append(event)
                
                # print文を出力
                print(f"⚡ [Frame {frame_number:04d}] Ball ID:{ball_id} - HIT DETECTED!")
                print(f"   Position: ({new_x}, {new_y})")
                print(f"   Speed change: {prev_speed:.1f} → {current_speed:.1f} px/frame")
                print(f"   X-Acceleration: {ax:.2f} px/frame² (gravity-independent)")
                print()
    
    def _detect_bounce(self, ball_id, trajectory, frame_number):
        """
        バウンドを検出（Y方向の速度反転）
        
        Args:
            ball_id: ボールID
            trajectory: 現在の軌跡
            frame_number: 現在のフレーム番号
        """
        # 最低3点必要
        if len(trajectory) < 3:
            return
        
        # 最新の3点を取得
        x1, y1, frame1, _, _ = trajectory[-3]
        x2, y2, frame2, _, _ = trajectory[-2]
        x3, y3, frame3, _, _ = trajectory[-1]
        
        dt1 = frame2 - frame1
        dt2 = frame3 - frame2
        
        if dt1 == 0 or dt2 == 0:
            return
        
        # Y方向の速度を計算
        vy1 = (y2 - y1) / dt1
        vy2 = (y3 - y2) / dt2
        
        # Y方向の速度が反転（下向き→上向き）= バウンド
        # vy1 > 0 (下向き), vy2 < 0 (上向き) ※画像座標系では下が+
        if vy1 > 1 and vy2 < -1:  # 閾値を設定して小さな変動を無視
            # イベント履歴を初期化（必要に応じて）
            if ball_id not in self.ball_events:
                self.ball_events[ball_id] = []
            
            # 最近のバウンドイベントと重複しないようにチェック
            recent_bounce = False
            for event in reversed(self.ball_events[ball_id]):
                if event['type'] == 'bounce' and (frame_number - event['frame']) < 5:
                    recent_bounce = True
                    break
            
            if not recent_bounce:
                event = {
                    'type': 'bounce',
                    'frame': frame_number,
                    'position': (x3, y3),
                    'vy_before': vy1,
                    'vy_after': vy2
                }
                self.ball_events[ball_id].append(event)
                
                # print文を出力
                print(f"🏓 [Frame {frame_number:04d}] Ball ID:{ball_id} - BOUNCE DETECTED!")
                print(f"   Position: ({x3}, {y3})")
                print(f"   Y-Velocity: {vy1:.1f} → {vy2:.1f} px/frame (reversed)")
                print()
        
    def predict_intersection(self, ball_id, robot_y, frame_width):
        """
        ボールがロボット位置（Y座標）と交差する点のX座標を予測
        
        Args:
            ball_id: ボールID
            robot_y: ロボットのY座標（画像上の位置）
            frame_width: フレームの幅
            
        Returns:
            (predicted_x, confidence, velocity_x, velocity_y) または None
        """
        if ball_id not in self.ball_trajectories:
            return None
        
        trajectory = self.ball_trajectories[ball_id]
        
        # 最低3点必要
        if len(trajectory) < 3:
            return None
        
        # 最新の数点を使用して速度を計算
        recent = trajectory[-3:]
        
        # 平均速度を計算
        velocities_x = []
        velocities_y = []
        
        for i in range(len(recent) - 1):
            x1, y1, frame1, _, _ = recent[i]
            x2, y2, frame2, _, _ = recent[i + 1]
            
            if frame2 != frame1:
                vx = (x2 - x1) / (frame2 - frame1)
                vy = (y2 - y1) / (frame2 - frame1)
                velocities_x.append(vx)
                velocities_y.append(vy)
        
        if not velocities_y:
            return None
        
        avg_vx = np.mean(velocities_x)
        avg_vy = np.mean(velocities_y)
        
        # Y方向の速度が0に近い場合は予測できない
        if abs(avg_vy) < 0.1:
            return None
        
        # 現在位置
        current_x, current_y, current_frame, _, _ = trajectory[-1]
        
        # ロボット位置に到達するまでのフレーム数を計算
        frames_to_robot = (robot_y - current_y) / avg_vy
        
        # 未来の予測のみ（ボールがロボットに向かっている場合）
        if frames_to_robot < 0:
            return None
        
        # X座標を予測
        predicted_x = current_x + avg_vx * frames_to_robot
        
        # 画面外の予測は信頼度を下げる
        confidence = 1.0
        if predicted_x < 0 or predicted_x > frame_width:
            confidence = 0.5
            predicted_x = max(0, min(frame_width, predicted_x))
        
        # 速度が安定しているかチェック（信頼度）
        if len(velocities_x) > 1:
            vx_std = np.std(velocities_x)
            vy_std = np.std(velocities_y)
            if vx_std > 10 or vy_std > 10:  # 速度の変動が大きい
                confidence *= 0.7
        
        return (int(predicted_x), confidence, avg_vx, avg_vy)
        
    def detect_balls(self, frame):
        """
        フレームから卓球ボールを検出
        
        Args:
            frame: 入力画像フレーム
            
        Returns:
            検出されたボールのリスト [(x, y, radius, color), ...]
        """
        # HSV色空間に変換
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # ガウシアンブラーでノイズ除去
        blurred = cv2.GaussianBlur(hsv, (11, 11), 0)
        
        detected_balls = []
        
        # 白ボールの検出
        if self.color_mode in ["white", "both"]:
            white_balls = self._detect_color(blurred, WHITE_LOWER, WHITE_UPPER, "white")
            detected_balls.extend(white_balls)
        
        # オレンジボールの検出
        if self.color_mode in ["orange", "both"]:
            orange_balls = self._detect_color(blurred, ORANGE_LOWER, ORANGE_UPPER, "orange")
            detected_balls.extend(orange_balls)
        
        return detected_balls
    
    def _detect_color(self, hsv_blurred, lower, upper, color_name):
        """
        特定の色のボールを検出
        
        Args:
            hsv_blurred: ブラー処理済みHSV画像
            lower: HSV下限値
            upper: HSV上限値
            color_name: 色の名前
            
        Returns:
            検出されたボールのリスト
        """
        # 色範囲でマスク作成
        mask = cv2.inRange(hsv_blurred, lower, upper)
        
        # モルフォロジー処理でノイズ除去
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 輪郭検出
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        balls = []
        
        for contour in contours:
            # 面積が小さすぎる輪郭は無視
            area = cv2.contourArea(contour)
            if area < 20:
                continue
            
            # 最小外接円を取得
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            # サイズチェック
            if radius < MIN_RADIUS or radius > MAX_RADIUS:
                continue
            
            # 円形度チェック
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity < MIN_CIRCULARITY:
                continue
            
            balls.append((int(x), int(y), int(radius), color_name))
        
        return balls
    
    def draw_detections(self, frame, balls, frame_number, robot_y):
        """
        検出結果を描画
        
        Args:
            frame: 描画対象のフレーム
            balls: 検出されたボールのリスト
            frame_number: 現在のフレーム番号
            robot_y: ロボットのY座標
        """
        height, width = frame.shape[:2]
        
        # ロボット位置のラインを描画
        cv2.line(frame, (0, robot_y), (width, robot_y), (0, 255, 255), 2)
        cv2.putText(
            frame,
            "Robot Position",
            (10, robot_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )
        
        # 軌跡を更新
        self.update_trajectories(balls, frame_number)
        
        # 各ボールの軌跡と予測を描画
        for ball_id, trajectory in self.ball_trajectories.items():
            if len(trajectory) == 0:
                continue
            
            # 最新の位置
            x, y, _, radius, color_name = trajectory[-1]
            
            # 最近のイベントをチェック（最近5フレーム以内）
            is_bounced = False
            is_hit = False
            
            if ball_id in self.ball_events:
                for event in reversed(self.ball_events[ball_id]):
                    if (frame_number - event['frame']) <= 5:
                        if event['type'] == 'bounce':
                            is_bounced = True
                        elif event['type'] == 'hit':
                            is_hit = True
            
            # イベントに応じて色を変更
            if is_hit:
                # 打球直後は赤色
                box_color = (0, 0, 255)  # 赤
                text_color = (255, 255, 255)  # 白
            elif is_bounced:
                # バウンド直後はマゼンタ/ピンク
                box_color = (255, 0, 255)  # マゼンタ
                text_color = (255, 255, 255)  # 白
            else:
                # 通常の色
                if color_name == "white":
                    box_color = (255, 255, 255)  # 白
                    text_color = (0, 0, 0)  # 黒（背景用）
                else:  # orange
                    box_color = (0, 165, 255)  # オレンジ
                    text_color = (255, 255, 255)  # 白
            
            # バウンディングボックス（正方形）を描画
            box_size = int(radius * 2.5)
            x1 = x - box_size // 2
            y1 = y - box_size // 2
            x2 = x + box_size // 2
            y2 = y + box_size // 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            # 円も描画（参考用）
            cv2.circle(frame, (x, y), radius, box_color, 2)
            
            # 中心点を描画
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            # 軌跡を描画
            if len(trajectory) > 1:
                for i in range(len(trajectory) - 1):
                    pt1 = (trajectory[i][0], trajectory[i][1])
                    pt2 = (trajectory[i + 1][0], trajectory[i + 1][1])
                    cv2.line(frame, pt1, pt2, box_color, 2)
            
            # 交点予測
            prediction = self.predict_intersection(ball_id, robot_y, width)
            
            if prediction is not None:
                pred_x, confidence, vx, vy = prediction
                
                # 予測点を描画
                cv2.circle(frame, (pred_x, robot_y), 10, (0, 0, 255), -1)
                cv2.circle(frame, (pred_x, robot_y), 15, (0, 0, 255), 2)
                
                # 予測軌道を点線で描画
                current_x, current_y = x, y
                steps = 20
                for i in range(steps):
                    t = i / steps
                    draw_x = int(current_x + vx * t * (robot_y - current_y) / vy)
                    draw_y = int(current_y + vy * t * (robot_y - current_y) / vy)
                    if 0 <= draw_x < width and 0 <= draw_y < height:
                        cv2.circle(frame, (draw_x, draw_y), 2, (0, 255, 0), -1)
                
                # 予測情報をテキスト表示
                pred_text = f"Target X: {pred_x}px (Conf: {confidence:.0%})"
                cv2.putText(
                    frame,
                    pred_text,
                    (10, robot_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )
                
                # 速度情報
                speed_text = f"Velocity: X={vx:.1f}, Y={vy:.1f} px/frame"
                cv2.putText(
                    frame,
                    speed_text,
                    (10, robot_y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
            
            # ラベルを描画
            label = f"{color_name.capitalize()} Ball (ID:{ball_id})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            # テキストサイズを取得
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            
            # テキスト背景を描画
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - 10),
                (x1 + text_width + 10, y1),
                box_color,
                -1
            )
            
            # テキストを描画
            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - 5),
                font,
                font_scale,
                text_color,
                thickness
            )
        
        return frame



# リサイズ後の幅と高さ（例: 640x360）
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 360

def process_video(video_path=None, output_path=None, color_mode="both"):
    """
    動画を処理してボールをトラッキング
    
    Args:
        video_path: 動画ファイルのパス（Noneでカメラ使用）
        output_path: 出力動画のパス（Noneで保存しない）
        color_mode: "white", "orange", "both"のいずれか
    """
    # ビデオキャプチャを開く
    if video_path is None:
        cap = cv2.VideoCapture(0)
        print("Using camera")
    else:
        cap = cv2.VideoCapture(video_path)
        print(f"Processing video: {video_path}")
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # 動画情報を取得
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # リサイズ後のサイズ
    width = RESIZE_WIDTH
    height = RESIZE_HEIGHT
    
    # ロボットのY位置を計算
    if ROBOT_Y_POSITION <= 1.0:
        robot_y = int(height * ROBOT_Y_POSITION)
    else:
        robot_y = int(ROBOT_Y_POSITION)
    
    print(f"Video info: {orig_width}x{orig_height} (resize to {width}x{height}), {fps} FPS, {total_frames} frames")
    print(f"Robot Y position: {robot_y}px ({(robot_y/height)*100:.1f}% from top)")
    
    # ビデオライターを初期化
    out = None
    if output_path is not None:
        # 出力ディレクトリを作成
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Output: {output_path}")
    
    # トラッカーを初期化
    tracker = BallTracker(color_mode, max_history=TRAJECTORY_HISTORY)
    
    frame_count = 0
    
    print("\nProcessing...")
    print("Press 'q' to quit, 'p' to pause")
    
    paused = False
    frame = None
    while True:
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                print("\nEnd of video or error reading frame")
                break
            
            # フレームをリサイズ
            frame = cv2.resize(frame, (width, height))
            
            frame_count += 1
            
            # ボールを検出
            balls = tracker.detect_balls(frame)
            
            # 検出結果を描画（軌跡と予測を含む）
            result_frame = tracker.draw_detections(frame.copy(), balls, frame_count, robot_y)
            
            # 情報を表示
            info_text = f"Frame: {frame_count}/{total_frames} | Balls: {len(balls)}"
            cv2.putText(
                result_frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # 結果を保存
            if out is not None:
                out.write(result_frame)
            
            # 進捗表示
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%) - Detected: {len(balls)} balls")
        else:
            if frame is not None:
                result_frame = frame.copy()
                cv2.putText(
                    result_frame,
                    "PAUSED - Press 'p' to resume",
                    (width // 2 - 150, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
            else:
                continue
        
        # 結果を表示
        cv2.imshow('Ball Tracking', result_frame)
        
        # キー入力処理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nStopped by user")
            break
        elif key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resumed")
    
    # クリーンアップ
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"\nProcessing complete!")
    print(f"Total frames processed: {frame_count}")
    if output_path is not None:
        print(f"Output saved to: {output_path}")


def main():
    """メイン処理"""
    # 動画ファイルの存在確認
    if VIDEO_PATH is not None and Path(VIDEO_PATH).exists():
        print(f"Video file found: {VIDEO_PATH}")
        process_video(VIDEO_PATH, OUTPUT_PATH, BALL_COLOR)
    elif VIDEO_PATH is None:
        print("Using camera mode")
        process_video(None, OUTPUT_PATH, BALL_COLOR)
    else:
        print(f"Video file not found: {VIDEO_PATH}")
        print("Switching to camera mode...")
        process_video(None, OUTPUT_PATH, BALL_COLOR)


if __name__ == "__main__":
    main()
