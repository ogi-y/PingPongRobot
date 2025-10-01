import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time

class ImageViewer(Node):
    def __init__(self):
        super().__init__('image_viewer')
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',  # 購読するトピック名
            self.image_callback,
            10)
        self.bridge = CvBridge()
        
        # 表示用パラメータ
        self.declare_parameter('display_scale', 1.0)
        self.declare_parameter('show_info', True)
        
        self.display_scale = self.get_parameter('display_scale').get_parameter_value().double_value
        self.show_info = self.get_parameter('show_info').get_parameter_value().bool_value
        
        self.frame_count = 0
        self.last_time = time.time()

    def image_callback(self, msg):
        # ROS画像メッセージ→OpenCV画像へ変換
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # 表示用にスケール調整
        if self.display_scale != 1.0:
            height, width = cv_image.shape[:2]
            new_width = int(width * self.display_scale)
            new_height = int(height * self.display_scale)
            cv_image = cv2.resize(cv_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # FPS計算とサイズ情報の表示
        if self.show_info:
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_time >= 1.0:  # 1秒ごとに更新
                fps = self.frame_count / (current_time - self.last_time)
                self.get_logger().info(f'表示FPS: {fps:.1f}, 画像サイズ: {cv_image.shape[1]}x{cv_image.shape[0]}')
                self.frame_count = 0
                self.last_time = current_time
        
        # 画像表示
        cv2.imshow('Received Image', cv_image)
        cv2.waitKey(1)  # ウィンドウを更新

def main(args=None):
    rclpy.init(args=args)
    node = ImageViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()