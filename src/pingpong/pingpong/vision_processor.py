import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import cv2
from deepface import DeepFace
from ultralytics import YOLO
import json

class VisionProcessor(Node):
    def __init__(self):
        super().__init__('vision_processor')
        self.sub_img = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.sub_age_trigger = self.create_subscription(Bool, '/age_trigger', self.age_trigger_callback, 10)
        self.sub_body_trigger = self.create_subscription(Bool, '/body_trigger', self.body_trigger_callback, 10)
        self.pub_age = self.create_publisher(String, '/age', 10)
        self.pub_player_pos = self.create_publisher(String, '/player_pos', 10)
        self.bridge = CvBridge()
        self.model_yolo = YOLO('yolov8n-pose.pt')

        self.latest_frame = None
        self.prev_age_trigger = False
        self.prev_body_trigger = False

        self.declare_parameter('left_threshold', 0.33)
        self.declare_parameter('right_threshold', 0.66)

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
        
        age_list = []
        for idx, result in enumerate(results, 1):
            if 'age' not in result:
                continue
            age_data = {"id": idx, "age": int(result['age'])}
            age_list.append(age_data)
        msg_data = {"ages": age_list}
        self.pub_age.publish(String(data=json.dumps(msg_data)))

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
        if not results or len(results[0].keypoints) == 0:
            self.get_logger().info('No bodies detected')
            return
        # ここでプレイヤーを識別　（今回は最初の人のみ）
        player_keypoints = results[0].keypoints.xy[0].cpu().numpy()
        h, w, _ = self.latest_frame.shape
        # 体の中心位置を計算 （今回は鼻の位置）
        player_x, player_y = player_keypoints[0]

        left_threshold = self.get_parameter('left_threshold').get_parameter_value().double_value
        right_threshold = self.get_parameter('right_threshold').get_parameter_value().double_value


        if player_x < w * left_threshold:
            position = "left"
        elif player_x > w * right_threshold:
            position = "right"
        else:
            position = "center"

        msg_data = {"pos": position, "x": int(player_x), "y": int(player_y)}
        self.pub_player_pos.publish(String(data=json.dumps(msg_data)))

def main():
    rclpy.init()
    node = VisionProcessor()
    executor = rclpy.executors.MultiThreadedExecutor()
    rclpy.spin(node, executor=executor)
    del node.model_yolo
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()