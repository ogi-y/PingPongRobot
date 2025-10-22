import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
from deepface import DeepFace

class ImageSizePublisher(Node):
    def __init__(self):
        super().__init__('image_size_publisher')
        self.sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.pub = self.create_publisher(String, 'age', 10)
        self.bridge = CvBridge()

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        results = DeepFace.analyze(cv_image, actions=['age'], enforce_detection=False)
        if not isinstance(results, list):
            results = [results]
        for idx, result in enumerate(results, 1):
            age = int(result['age'])
            msg_str = f"id:{idx}, age:{age}"
            self.get_logger().info(msg_str)
            self.pub.publish(String(data=msg_str))

def main():
    rclpy.init()
    node = ImageSizePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()