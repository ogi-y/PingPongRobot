import cv2
import numpy as np
from collections import deque

class RobustPingPongTracker:
    def __init__(self, buffer_size=32):
        # オレンジ色の卓球ボール用HSV範囲
        self.lower_orange = np.array([5, 100, 100])
        self.upper_orange = np.array([25, 255, 255])
        
        # 白色ボール用の範囲
        self.lower_white = np.array([0, 0, 200])
        self.upper_white = np.array([180, 30, 255])
        
        self.pts = deque(maxlen=buffer_size)
        self.use_orange = True
        
        # カルマンフィルタ
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                   [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                  [0, 1, 0, 1],
                                                  [0, 0, 1, 0],
                                                  [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.prediction = None
        
        # 前フレームの情報
        self.prev_center = None
        self.prev_radius = None
        self.lost_frames = 0
        
        # ハフ変換用パラメータ
        self.use_hough = True
        
    def is_circular(self, contour):
        """輪郭が円形かどうかを判定"""
        area = cv2.contourArea(contour)
        if area < 10:
            return False, 0
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False, 0
        
        # 円形度を計算 (1に近いほど円形)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        return circularity > 0.6, circularity
    
    def motion_consistency(self, center):
        """動きの一貫性をチェック"""
        if self.prev_center is None or len(self.pts) < 2:
            return True
        
        # 前フレームからの移動距離
        dx = center[0] - self.prev_center[0]
        dy = center[1] - self.prev_center[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # 急激な移動は除外（画面幅の30%以上の移動）
        if distance > 640 * 0.3:
            return False
        
        return True
    
    def size_consistency(self, radius):
        """サイズの一貫性をチェック"""
        if self.prev_radius is None:
            return True
        
        # 前フレームとのサイズ比較（50%以上の変化は除外）
        ratio = radius / self.prev_radius if self.prev_radius > 0 else 1
        if ratio < 0.5 or ratio > 2.0:
            return False
        
        return True
    
    def detect_with_contours(self, mask):
        """輪郭ベースの検出"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        
        for contour in contours:
            # 円形度チェック
            is_circ, circularity = self.is_circular(contour)
            if not is_circ:
                continue
            
            ((x, y), r) = cv2.minEnclosingCircle(contour)
            
            # サイズフィルタ
            if r < 3 or r > 150:
                continue
            
            center = (int(x * 2), int(y * 2))
            radius = int(r * 2)
            
            # 動きの一貫性チェック
            if not self.motion_consistency(center):
                continue
            
            # サイズの一貫性チェック
            if not self.size_consistency(radius):
                continue
            
            # スコア計算（円形度、サイズ、予測位置との距離）
            score = circularity
            
            # 予測位置との距離を考慮
            if self.prediction is not None:
                pred_x = int(self.prediction[0])
                pred_y = int(self.prediction[1])
                dist = np.sqrt((center[0] - pred_x)**2 + (center[1] - pred_y)**2)
                # 距離が近いほどスコアが高い
                dist_score = 1.0 / (1.0 + dist / 100.0)
                score = circularity * 0.5 + dist_score * 0.5
            
            candidates.append((center, radius, score, cv2.contourArea(contour)))
        
        return candidates
    
    def detect_with_hough(self, mask, gray):
        """ハフ変換による円検出"""
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=75
        )
        
        candidates = []
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, r = circle
                center = (int(x * 2), int(y * 2))
                radius = int(r * 2)
                
                # 動きとサイズの一貫性チェック
                if not self.motion_consistency(center):
                    continue
                if not self.size_consistency(radius):
                    continue
                
                # マスク内で検出されているかチェック
                mask_value = mask[min(y, mask.shape[0]-1), 
                                 min(x, mask.shape[1]-1)]
                if mask_value > 0:
                    score = 0.8  # ハフ変換での検出はやや低めのスコア
                    candidates.append((center, radius, score, np.pi * r * r))
        
        return candidates
    
    def process_frame(self, frame):
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        
        # ガウシアンぼかし
        blurred = cv2.GaussianBlur(small, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        
        # マスク作成
        if self.use_orange:
            mask = cv2.inRange(hsv, self.lower_orange, self.upper_orange)
        else:
            mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        
        # モルフォロジー処理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 輪郭ベースの検出
        contour_candidates = self.detect_with_contours(mask)
        
        # ハフ変換による検出（オプション）
        hough_candidates = []
        if self.use_hough and len(contour_candidates) < 3:
            hough_candidates = self.detect_with_hough(mask, gray)
        
        # 全候補を統合
        all_candidates = contour_candidates + hough_candidates
        
        center = None
        radius = 0
        
        if all_candidates:
            # スコアが最も高い候補を選択
            all_candidates.sort(key=lambda x: x[2], reverse=True)
            center, radius, score, area = all_candidates[0]
            
            # カルマンフィルタで予測
            measurement = np.array([[np.float32(center[0])],
                                   [np.float32(center[1])]])
            self.kalman.correct(measurement)
            self.prediction = self.kalman.predict()
            
            self.pts.appendleft(center)
            self.prev_center = center
            self.prev_radius = radius
            self.lost_frames = 0
            
        elif self.prediction is not None and self.lost_frames < 10:
            # 検出できない場合は予測位置を使用（最大10フレーム）
            pred_x = int(self.prediction[0])
            pred_y = int(self.prediction[1])
            center = (pred_x, pred_y)
            self.prediction = self.kalman.predict()
            self.lost_frames += 1
        else:
            self.lost_frames += 1
            if self.lost_frames > 30:
                # 長時間見失った場合はリセット
                self.prev_center = None
                self.prev_radius = None
                self.prediction = None
        
        return center, radius, mask, len(all_candidates)
    
    def draw_tracking(self, frame, center, radius, num_candidates):
        if center is not None and radius > 0:
            # ボールの円を描画
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            cv2.circle(frame, center, 3, (0, 0, 255), -1)
            
            # 信頼度表示
            confidence = max(0, 100 - self.lost_frames * 10)
            cv2.putText(frame, f"Conf: {confidence}%", 
                       (center[0] - 30, center[1] - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 軌跡を描画
        for i in range(1, len(self.pts)):
            if self.pts[i - 1] is None or self.pts[i] is None:
                continue
            thickness = int(np.sqrt(32 / float(i + 1)) * 1.5)
            cv2.line(frame, self.pts[i - 1], self.pts[i], 
                    (0, 255, 255), thickness)
        
        # 候補数表示
        cv2.putText(frame, f"Candidates: {num_candidates}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

def main():
    cap = cv2.VideoCapture(0)
    
    # 高速化設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    tracker = RobustPingPongTracker(buffer_size=32)
    
    fps_counter = deque(maxlen=30)
    
    print("頑健な卓球ボールトラッキング開始")
    print("'o': オレンジボール / 'w': 白ボール")
    print("'h': ハフ変換 ON/OFF / 'q': 終了")
    
    while True:
        start_time = cv2.getTickCount()
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # トラッキング処理
        center, radius, mask, num_candidates = tracker.process_frame(frame)
        
        # 描画
        frame = tracker.draw_tracking(frame, center, radius, num_candidates)
        
        # FPS計算
        elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        fps = 1.0 / elapsed if elapsed > 0 else 0
        fps_counter.append(fps)
        avg_fps = np.mean(fps_counter)
        
        # 情報表示
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        mode = "ORANGE" if tracker.use_orange else "WHITE"
        cv2.putText(frame, f"Mode: {mode}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        hough_status = "ON" if tracker.use_hough else "OFF"
        cv2.putText(frame, f"Hough: {hough_status}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # 結果表示
        cv2.imshow("Robust Ping Pong Tracking", frame)
        cv2.imshow("Mask", cv2.resize(mask, (320, 240)))
        
        # キー入力
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('o'):
            tracker.use_orange = True
            print("オレンジボールモードに切り替え")
        elif key == ord('w'):
            tracker.use_orange = False
            print("白ボールモードに切り替え")
        elif key == ord('h'):
            tracker.use_hough = not tracker.use_hough
            status = "ON" if tracker.use_hough else "OFF"
            print(f"ハフ変換: {status}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()