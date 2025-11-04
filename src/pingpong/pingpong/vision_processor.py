import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import cv2
from deepface import DeepFace
from ultralytics import YOLO

class VisionProcessor(Node):
    def __init__(self):
        super().__init__('vision_processor')
        self.sub_img = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.sub_age_trigger = self.create_subscription(Bool, '/age_trigger', self.age_trigger_callback, 10)
        self.sub_body_trigger = self.create_subscription(Bool, '/body_trigger', self.body_trigger_callback, 10)
        self.pub_age = self.create_publisher(String, '/age', 10)
        self.pub_body = self.create_publisher(String, '/body', 10)
        self.bridge = CvBridge()
        self.model_yolo = YOLO('yolov8n.pt')

        self.latest_frame = None
        self.prev_age_trigger = False
        self.prev_body_trigger = False
        self.get_logger().info('Vision Processor Node Initialized')

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.latest_frame = cv_image

    def age_trigger_callback(self, msg):
        if self.prev_age_trigger != msg.data:
            self.get_logger().info(f'Age trigger changed: {msg.data}')
            self.prev_age_trigger = msg.data

        if not msg.data:
            return
        
        if self.latest_frame is None:
            self.get_logger().warn('No image frame available for age estimation')
            return
        
        results = DeepFace.analyze(self.latest_frame, actions=['age'], enforce_detection=False)
        if not isinstance(results, list):
            results = [results]
        for idx, result in enumerate(results, 1):
            age = int(result['age'])
            msg_str = f"id:{idx}, age:{age}"
            # self.get_logger().info(msg_str)
            self.pub_age.publish(String(data=msg_str))
        
    def body_trigger_callback(self, msg):
        if self.prev_body_trigger != msg.data:
            self.get_logger().info(f'Body trigger changed: {msg.data}')
            self.prev_body_trigger = msg.data
            
        if not msg.data:
            return
        
        if self.latest_frame is None:
            self.get_logger().warn('No image frame available for body detection')
            return
        
        results = self.model_yolo(self.latest_frame, verbose=False)
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                msg_str = f"class_id:{cls_id}, confidence:{conf:.2f}"
                self.pub_body.publish(String(data=msg_str))

def main():
    rclpy.init()
    node = VisionProcessor()
    executor = rclpy.executors.MultiThreadedExecutor()
    rclpy.spin(node, executor=executor)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()