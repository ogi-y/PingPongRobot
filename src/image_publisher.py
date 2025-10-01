import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
import os

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')
        self.publisher_ = self.create_publisher(Image, '/image_raw', 10)
        self.bridge = CvBridge()
        
        # ROSパラメータの宣言と取得
        self.declare_parameter('target_width', 1280)
        self.declare_parameter('target_height', 720)
        self.declare_parameter('resize_enabled', True)
        
        self.target_width = self.get_parameter('target_width').get_parameter_value().integer_value
        self.target_height = self.get_parameter('target_height').get_parameter_value().integer_value
        self.resize_enabled = self.get_parameter('resize_enabled').get_parameter_value().bool_value
        
        # 送信したい画像ファイルのパス
        self.image_path = os.path.join(os.path.dirname(__file__), '../data/pic/pic.jpg')
        self.timer = self.create_timer(1.0, self.timer_callback)  # 1秒ごと
        
        # 画像の読み込みとリサイズ
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            self.get_logger().error(f"画像ファイルが見つかりません: {self.image_path}")
            self.processed_image = None
        else:
            self.processed_image = self.prepare_image()

    def prepare_image(self):
        """画像の前処理（リサイズ）を行う"""
        if self.original_image is None:
            return None
            
        original_height, original_width = self.original_image.shape[:2]
        self.get_logger().info(f"元画像サイズ: {original_width}x{original_height}")
        
        if not self.resize_enabled:
            self.get_logger().info("リサイズは無効です - 元画像をそのまま使用")
            return self.original_image
            
        # アスペクト比を保持しながらリサイズ
        aspect_ratio = original_width / original_height
        target_aspect_ratio = self.target_width / self.target_height
        
        if aspect_ratio > target_aspect_ratio:
            # 横長の画像：幅を基準にリサイズ
            new_width = self.target_width
            new_height = int(self.target_width / aspect_ratio)
        else:
            # 縦長の画像：高さを基準にリサイズ
            new_height = self.target_height
            new_width = int(self.target_height * aspect_ratio)
        
        resized_image = cv2.resize(self.original_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        self.get_logger().info(f"リサイズ後サイズ: {new_width}x{new_height}")
        
        # データサイズの削減率を計算
        original_size = original_width * original_height * 3  # BGR 3チャンネル
        resized_size = new_width * new_height * 3
        reduction_rate = (1 - resized_size / original_size) * 100
        self.get_logger().info(f"データサイズ削減率: {reduction_rate:.1f}%")
        
        return resized_image

    def timer_callback(self):
        if self.processed_image is not None:
            msg = self.bridge.cv2_to_imgmsg(self.processed_image, encoding='bgr8')
            self.publisher_.publish(msg)
            height, width = self.processed_image.shape[:2]
            self.get_logger().info(f'画像をPublishしました ({width}x{height})')

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()