import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import threading
import cv2
import json
from ultralytics import YOLO
from deepface import DeepFace

class VisionAnalyzer(Node):
    def __init__(self):
        super().__init__('vision_analyzer')
        
        # --- 設定 ---
        self.declare_parameter('source', '0') # ノード実行時　ros2 run py_pingpong analyzer --ros-args -p source:='ウェブカメラのアドレス or USBカメラID'
        source_param = self.get_parameter('source').get_parameter_value().string_value
        if source_param.isdigit():
            self.source = int(source_param)
        else:
            self.source = source_param
        
        # ラケットを持つ手
        self.declare_parameter('hand', 'right')
        self.hand_side = self.get_parameter('hand').get_parameter_value().string_value

        # --- カメラ初期化 ---
        self.get_logger().info(f"Opening camera source: {self.source}")
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)# ウェブカメラのときはffmpeg
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # バッファを最小に

        if not self.cap.isOpened():
            self.get_logger().error("Could not open camera!")
            exit()

        # --- 変数初期化 ---
        self.latest_frame = None       # AI処理用
        self.current_raw_frame = None  # カメラ取り込み用
        self.lock = threading.Lock()   # 排他制御
        self.running = True            # スレッド制御用フラグ
        
        self.current_age = "Estimating..."
        self.frame_count = 0

        # --- モデル読み込み ---
        self.get_logger().info("Loading YOLOv8 Pose model...")
        self.pose_model = YOLO('yolov8n-pose.pt')
        self.pose_model.to('cpu')

        # --- ROS2設定 ---
        self.callback_group = ReentrantCallbackGroup()
        self.pub_image = self.create_publisher(Image, '/camera/image_raw', 10)
        self.pub_result = self.create_publisher(String, '/vision/analysis', 10)
        self.bridge = CvBridge()

        # --- スレッド開始（ここが重要！） ---
        # カメラ映像をひたすら取り込む専用のスレッドを起動
        self.capture_thread = threading.Thread(target=self.update_camera_buffer)
        self.capture_thread.daemon = True
        self.capture_thread.start()

        # AI処理用のタイマー
        self.timer = self.create_timer(0.01, self.process_ai, callback_group=self.callback_group)
        self.timer_age = self.create_timer(1.0, self.process_age, callback_group=self.callback_group)

    # --- 【追加】カメラ映像を常に最新にする関数 ---
    def update_camera_buffer(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    # バッファに溜めず、常に最新画像で上書きする
                    self.current_raw_frame = frame
            else:
                # 読み込み失敗時は少し待つ（CPU空回り防止）
                cv2.waitKey(10)

    # --- メイン処理 ---
    def process_ai(self):
        # 最新の画像を取得
        frame = None
        with self.lock:
            if self.current_raw_frame is not None:
                frame = self.current_raw_frame.copy()
        
        if frame is None:
            return

        # リサイズ（負荷軽減）
        frame = cv2.resize(frame, (640, 480))
        
        # DeepFace用にも保存
        self.latest_frame = frame 

        # YOLO推論
        results = self.pose_model(frame, verbose=False)
        annotated_frame = results[0].plot()

        # 人数カウントなどのロジック
        detected_people = []
        if results[0].keypoints is not None and results[0].keypoints.data.shape[1] > 0:
            keypoints = results[0].keypoints.data.cpu().numpy()
            for i, person_kpts in enumerate(keypoints):
                if len(person_kpts) > 0:
                    
                    # 鼻の座標（予備用）
                    nose_x, nose_y, nose_conf = person_kpts[0]

                    # 指定された手首の座標を取得
                    if self.hand_side == 'left':
                        hand_x, hand_y, hand_conf = person_kpts[9]  # 左手首
                    else:
                        hand_x, hand_y, hand_conf = person_kpts[10] # 右手首

                    # --- ロジック: 手が見えていれば手、見えなければ鼻を使う ---
                    if hand_conf > 0.5:
                        final_x, final_y = hand_x, hand_y
                        body_part = "Hand"
                    else:
                        final_x, final_y = nose_x, nose_y
                        body_part = "Nose (Fallback)"

                    detected_people.append({
                        "id": i, 
                        "pos_x": float(final_x), 
                        "pos_y": float(final_y),
                        "part_used": body_part, # デバッグ用に何を使ったか記録
                        "pose_detected": True
                    })
                    
                    # 画面にも描画（赤い丸を手首に描く）
                    if hand_conf > 0.5:
                        cv2.circle(annotated_frame, (int(hand_x), int(hand_y)), 10, (0, 0, 255), -1)

        # 年齢描画
        cv2.putText(annotated_frame, f"Age: {self.current_age}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Publish
        try:
            ros_image = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
            self.pub_image.publish(ros_image)
            
            analysis_data = {
                "age": self.current_age,
                "people": detected_people
            }
            msg = String()
            msg.data = json.dumps(analysis_data)
            self.pub_result.publish(msg)
        except Exception:
            pass

    def process_age(self):
        # (変更なし) DeepFace処理...
        frame_to_process = None
        if self.latest_frame is not None:
            frame_to_process = self.latest_frame.copy()
            
        if frame_to_process is not None:
            try:
                analysis = DeepFace.analyze(frame_to_process, actions=['age'], enforce_detection=False)
                if isinstance(analysis, list) and len(analysis) > 0:
                    self.current_age = str(analysis[0]['age'])
                elif isinstance(analysis, dict):
                    self.current_age = str(analysis['age'])
            except Exception:
                pass

    def __del__(self):
        self.running = False
        if hasattr(self, 'capture_thread') and self.capture_thread.is_alive():
            self.capture_thread.join()
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

def main(args=None):
    rclpy.init(args=args)
    node = VisionAnalyzer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        executor.shutdown()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except:
            pass
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()