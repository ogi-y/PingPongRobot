import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageViewer(Node):
    def __init__(self):
        super().__init__('image_viewer')
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',  # 購読するトピック名
            self.image_callback,
            10)
        self.bridge = CvBridge()

    def image_callback(self, msg):
        # ROS画像メッセージ→OpenCV画像へ変換
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
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