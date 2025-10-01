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
        # 送信したい画像ファイルのパス
        self.image_path = os.path.join(os.path.dirname(__file__), '../data/pic/pic.jpg')
        self.timer = self.create_timer(1.0, self.timer_callback)  # 1秒ごと
        self.cv_image = cv2.imread(self.image_path)
        if self.cv_image is None:
            self.get_logger().error(f"画像ファイルが見つかりません: {self.image_path}")

    def timer_callback(self):
        if self.cv_image is not None:
            msg = self.bridge.cv2_to_imgmsg(self.cv_image, encoding='bgr8')
            self.publisher_.publish(msg)
            self.get_logger().info('画像をPublishしました')

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()