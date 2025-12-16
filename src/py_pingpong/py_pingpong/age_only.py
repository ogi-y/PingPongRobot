import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import json
from deepface import DeepFace

class AgeEstimatorNode(Node):
    def __init__(self):
        super().__init__('age_estimator')

        # --- 設定 ---
        self.declare_parameter('source', '0')
        source_param = self.get_parameter('source').get_parameter_value().string_value
        self.source = int(source_param) if source_param.isdigit() else source_param

        # --- カメラ初期化 ---
        self.get_logger().info(f"Opening camera source: {self.source}")
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            self.get_logger().error("Could not open camera!")
            exit()

        # --- 変数初期化 ---
        self.latest_frame = None  # AI処理用
        self.current_age = "?"    # 推定結果表示用

        # --- ROS2設定 ---
        self.bridge = CvBridge()
        self.callback_group = ReentrantCallbackGroup() # 並行処理を許可

        # Publisher
        self.pub_image = self.create_publisher(Image, '/camera/image_raw', 10)
        self.pub_age = self.create_publisher(String, '/vision/age_data', 10)

        # --- タイマー設定 ---
        # 1. 映像表示・配信用（高速: 30fps）
        self.timer_display = self.create_timer(0.033, self.process_display, callback_group=self.callback_group)
        
        # 2. 年齢推定用（低速: 1fps程度 ※DeepFaceは重いため頻度を下げる）
        self.timer_ai = self.create_timer(1.0, self.process_age, callback_group=self.callback_group)

    def process_display(self):
        """カメラ画像を取得し、現在の推定年齢を描画して配信する"""
        ret, frame = self.cap.read()
        if not ret:
            return

        # AIスレッド用に最新フレームをコピーして保存
        self.latest_frame = frame.copy()

        # リサイズ（通信負荷軽減のため）
        display_frame = cv2.resize(frame, (640, 480))

        # 結果の描画（AI処理は待たずに、今の変数の値を描画）
        text = f"Estimated Age: {self.current_age}"
        cv2.putText(display_frame, text, (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # ROSメッセージに変換してPublish
        try:
            msg = self.bridge.cv2_to_imgmsg(display_frame, "bgr8")
            self.pub_image.publish(msg)
        except Exception as e:
            self.get_logger().warn(f"Publish error: {e}")

    def process_age(self):
        """最新フレームを使って年齢を推定する（重い処理）"""
        if self.latest_frame is None:
            return

        try:
            # DeepFaceで年齢のみ解析 (enforce_detection=Falseで顔が見つからなくてもエラーにしない)
            # ※ actions=['age'] を指定して軽量化
            analysis = DeepFace.analyze(
                img_path=self.latest_frame, 
                actions=['age'], 
                enforce_detection=False, 
                detector_backend='opencv', # 軽量なバックエンドを指定
                silent=True
            )

            # 結果の取り出し（リストまたは辞書形式に対応）
            result = analysis[0] if isinstance(analysis, list) else analysis
            
            # 年齢を更新
            age_val = result.get('age', 0)
            if age_val > 0:
                self.current_age = str(age_val)
                
                # 結果をPublish
                json_msg = String()
                json_msg.data = json.dumps({"age": int(age_val)})
                self.pub_age.publish(json_msg)
                
                self.get_logger().info(f"Updated Age: {self.current_age}")

        except Exception as e:
            # 顔が見つからない場合などはここに来る
            pass

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

def main(args=None):
    rclpy.init(args=args)
    node = AgeEstimatorNode()
    
    # AI処理中に映像を止めないために MultiThreadedExecutor を使用
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()