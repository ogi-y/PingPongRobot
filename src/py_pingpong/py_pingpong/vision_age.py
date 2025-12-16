# vision_age_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy
from cv_bridge import CvBridge
from deepface import DeepFace
import cv2

class VisionAgeNode(Node):
    def __init__(self):
        super().__init__('vision_age_node')
        
        self.bridge = CvBridge()
        
        # --- 重要: QoS設定 ---
        qos_sensor_sub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # YOLOノードから来る画像を購読
        self.sub_image = self.create_subscription(
            Image, 
            '/vision/image_for_age', 
            self.image_callback, 
            qos_profile=qos_sensor_sub)

        # 解析結果をPublish
        self.pub_result = self.create_publisher(String, '/vision/age_result', 10)
        
        self.get_logger().info("Age Estimation Node Ready. Waiting for images...")
        # 初回モデルロードを済ませておく（最初の処理が遅くなるのを防ぐため）
        try:
            # ダミー画像で一度実行
            dummy_img = cv2.np.zeros((100, 100, 3), dtype='uint8')
            DeepFace.analyze(dummy_img, actions=['age'], enforce_detection=False)
            self.get_logger().info("DeepFace model loaded.")
        except Exception:
             self.get_logger().info("Model load triggered, will complete on first image.")

    def image_callback(self, msg):
        # 画像が届いたら実行される。この処理中は次の画像が来てもQoS設定により破棄される。
        try:
            # ROS ImageメッセージをOpenCV形式に変換
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # --- DeepFace実行 (重い処理) ---
            # enforce_detection=False は顔が見つからなくてもエラーにしない設定
            analysis = DeepFace.analyze(cv_image, actions=['age'], enforce_detection=False)
            
            age_str = "Unknown"
            if isinstance(analysis, list) and len(analysis) > 0:
                age_str = str(analysis[0]['age'])
            elif isinstance(analysis, dict):
                age_str = str(analysis['age'])
            
            # 結果をPublish
            res_msg = String()
            res_msg.data = age_str
            self.pub_result.publish(res_msg)
            
            # デバッグ用ログ（本番ではコメントアウトしても良い）
            # self.get_logger().info(f"Estimated Age: {age_str}")
            
        except Exception as e:
            self.get_logger().warn(f"Age estimation failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = VisionAgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except:
            pass

if __name__ == '__main__':
    main()