import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class FaceDetector(Node):
    def __init__(self):
        super().__init__('face_detector')
        self.subscription = self.create_subscription(
            Image, 'input_image', self.listener_callback, 10)
        self.publisher_ = self.create_publisher(Image, 'face_image', 10)
        self.bridge = CvBridge()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def listener_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_img = cv_image[y:y+h, x:x+w]
            face_msg = self.bridge.cv2_to_imgmsg(face_img, encoding='bgr8')
            self.publisher_.publish(face_msg)
            break  # 1人目だけ

def main(args=None):
    rclpy.init(args=args)
    node = FaceDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()