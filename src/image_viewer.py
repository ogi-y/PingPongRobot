import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')
        self.publisher_ = self.create_publisher(Image, 'input_image', 10)
        self.bridge = CvBridge()
        # プロジェクトディレクトリ基準で画像パスを指定
        self.image_path = os.path.join(os.path.dirname(__file__), '../../pic/pic/sample.jpg')

        # 1秒ごとに画像をpublish
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        cv_image = cv2.imread(self.image_path)
        if cv_image is None:
            self.get_logger().error(f'画像が見つかりません: {self.image_path}')
            return
        msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        self.publisher_.publish(msg)
        self.get_logger().info('画像をpublishしました')

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()