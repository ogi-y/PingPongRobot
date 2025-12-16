# vision_yolo_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy
from cv_bridge import CvBridge
import threading
import cv2
import json
from ultralytics import YOLO

class VisionYoloNode(Node):
    def __init__(self):
        super().__init__('vision_yolo_node')
        
        # --- 設定 ---
        self.declare_parameter('source', '0')
        source_param = self.get_parameter('source').get_parameter_value().string_value
        self.source = int(source_param) if source_param.isdigit() else source_param
        
        self.declare_parameter('hand', 'right')
        self.hand_side = self.get_parameter('hand').get_parameter_value().string_value

        # --- カメラ初期化 ---
        self.get_logger().info(f"Opening camera source: {self.source}")
        self.cap = cv2.VideoCapture(self.source)
        # V4L2 backend使用時はバッファ設定が有効な場合がある
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            self.get_logger().error("Could not open camera!")
            exit()

        # --- 変数初期化 ---
        self.current_raw_frame = None  # カメラ取り込み用
        self.lock = threading.Lock()   # 排他制御
        self.running = True            # スレッド制御用フラグ
        
        # 年齢表示用変数（初期値）
        self.current_age_display = "Estimating..."

        # --- モデル読み込み ---
        self.get_logger().info("Loading YOLOv8 Pose model...")
        self.pose_model = YOLO('yolov8n-pose.pt')
        # GPUがあるなら 'cuda' に変更推奨
        self.pose_model.to('cpu') 

        # --- ROS2通信設定 ---
        self.bridge = CvBridge()
        
        # 1. YOLO結果画像とデータのPublish（既存）
        self.pub_image = self.create_publisher(Image, '/camera/image_raw', 10)
        self.pub_result = self.create_publisher(String, '/vision/analysis', 10)

        # 2. 【追加】年齢推定ノードへ画像を送るPublisher
        # 相手が処理しきれない分は捨てて良いので、ReliabilityはBEST_EFFORTにする
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.pub_age_request = self.create_publisher(Image, '/vision/image_for_age', qos_sensor)

        # 3. 【追加】年齢推定結果を受け取るSubscription
        self.create_subscription(String, '/vision/age_result', self.age_callback, 10)

        # --- スレッド開始 ---
        # カメラ映像キャプチャスレッド
        self.capture_thread = threading.Thread(target=self.update_camera_buffer)
        self.capture_thread.daemon = True
        self.capture_thread.start()

        # メイン処理タイマー (YOLO推論用)
        # 別プロセスにしたので MultiThreadedExecutor は不要になり、通常のタイマー実行でOK
        self.timer = self.create_timer(0.01, self.process_ai)

    # --- カメラ映像を常に最新にするスレッド関数 ---
    def update_camera_buffer(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.current_raw_frame = frame
            else:
                cv2.waitKey(10)

    # --- 【追加】年齢推定結果を受け取るコールバック ---
    def age_callback(self, msg):
        # 別ノードのDeepFace処理が終わるとこれが呼ばれる
        # メインループを止めずに変数を更新するだけ
        self.current_age_display = msg.data

    # --- メイン処理 ---
    def process_ai(self):
        # 最新の画像を取得
        frame = None
        with self.lock:
            if self.current_raw_frame is not None:
                frame = self.current_raw_frame.copy()
        
        if frame is None:
            return

        # リサイズ（YOLO負荷軽減のため）
        frame_resized = cv2.resize(frame, (640, 480))
        
        # --- 【追加】年齢推定ノードへ画像を送信 ---
        # YOLOの処理とは非同期に行われるため、ここで送信してもYOLOは止まらない
        try:
            # 通信負荷削減のためリサイズ後の画像を送る
            ros_image_for_age = self.bridge.cv2_to_imgmsg(frame_resized, "bgr8")
            self.pub_age_request.publish(ros_image_for_age)
        except Exception as e:
            self.get_logger().warn(f"Failed to publish image for age estimation: {e}")

        # --- YOLO推論 (元のロジック維持) ---
        results = self.pose_model(frame_resized, verbose=False)
        annotated_frame = results[0].plot()

        detected_people = []
        if results[0].keypoints is not None and results[0].keypoints.data.shape[1] > 0:
            keypoints = results[0].keypoints.data.cpu().numpy()
            for i, person_kpts in enumerate(keypoints):
                if len(person_kpts) > 0:
                    nose_x, nose_y, nose_conf = person_kpts[0]
                    
                    if self.hand_side == 'left':
                        hand_x, hand_y, hand_conf = person_kpts[9]
                    else:
                        hand_x, hand_y, hand_conf = person_kpts[10]

                    if hand_conf > 0.5:
                        final_x, final_y = hand_x, hand_y
                        body_part = "Hand"
                    else:
                        final_x, final_y = nose_x, nose_y
                        body_part = "Nose (Fallback)"

                    detected_people.append({
                        "id": i, "pos_x": float(final_x), "pos_y": float(final_y),
                        "part_used": body_part, "pose_detected": True
                    })
                    
                    if hand_conf > 0.5:
                        cv2.circle(annotated_frame, (int(hand_x), int(hand_y)), 10, (0, 0, 255), -1)

        # --- 年齢描画 (非同期で更新された変数を表示) ---
        cv2.putText(annotated_frame, f"Age: {self.current_age_display}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # --- 結果Publish ---
        try:
            ros_image = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
            self.pub_image.publish(ros_image)
            
            analysis_data = {
                # jsonには数値を入れたい場合、文字列から変換が必要かもしれません
                "age_str": self.current_age_display, 
                "people": detected_people
            }
            msg = String()
            msg.data = json.dumps(analysis_data)
            self.pub_result.publish(msg)
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
    node = VisionYoloNode()
    # 単一スレッドで十分なため、通常のspinを使用
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        node.destroy_node()
        try:
            rclpy.shutdown()
        except:
            pass
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()