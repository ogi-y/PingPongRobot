import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2

class ImageSizePublisher(Node):
    def __init__(self):
        super().__init__('image_size_publisher')
        self.sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.pub = self.create_publisher(String, 'image_size', 10)
        self.bridge = CvBridge()

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        #ここで画像処理ができる
        #gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        size_str = f"{cv_image.shape[1]}x{cv_image.shape[0]}"
        self.pub.publish(String(data=size_str))
        self.get_logger().info(f'Published image size: {size_str}')

def main():
    rclpy.init()
    node = ImageSizePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()