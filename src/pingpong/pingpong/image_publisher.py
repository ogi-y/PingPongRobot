import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import glob
import os
from pathlib import Path

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')
        self.pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()

        # 画像フォルダを指定
        data_dir = Path.home() / 'PingPongRobot' / 'data' / 'images'
        self.image_files = sorted(data_dir.glob('*.*'))

        if not self.image_files:
            self.get_logger().warn(f'No images found in {data_dir}. Create data/images and add some jpg/png.')
        else:
            self.get_logger().info(f'Found {len(self.image_files)} images in {data_dir}')

        self.idx = 0
        # 投稿頻度の設定
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        if not self.image_files:
            return
        img_path = self.image_files[self.idx]
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (640, 480))
        if img is None:
            self.get_logger().warn(f'Failed to read image: {img_path}')
            self.idx = (self.idx + 1) % len(self.image_files)
            return

        msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        self.pub.publish(msg)
        self.get_logger().info(f'Published {os.path.basename(img_path)}')
        self.idx = (self.idx + 1) % len(self.image_files)

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
