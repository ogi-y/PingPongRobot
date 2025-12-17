import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import json
from ultralytics import YOLO

class PoseEstimationNode(Node):
    def __init__(self):
        super().__init__('pose_estimation_node')
        
        self.declare_parameter('hand_side', 'right')
        self.declare_parameter('model_path', 'yolov8n-pose.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('process_interval', 1)
        
        self.hand_side = self.get_parameter('hand_side').get_parameter_value().string_value
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        
        self.bridge = CvBridge()
        self.pose_model = YOLO(model_path)
        self.frame_count = 0
        
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        self.pose_pub = self.create_publisher(String, '/pose/detected_positions', 10)
        self.annotated_image_pub = self.create_publisher(Image, '/pose/annotated_image', 10)
        
        self.get_logger().info(f'Pose estimation node started. Hand side: {self.hand_side}')
    
    def image_callback(self, msg):
        self.frame_count += 1
        if self.frame_count % self.process_interval != 0:
            return
        
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        results = self.pose_model(frame, verbose=False)
        annotated_frame = results[0].plot()
        
        detected_people = []
        
        if results[0].keypoints is not None and results[0].keypoints.data.shape[1] > 0:
            keypoints = results[0].keypoints.data.cpu().numpy()
            
            for i, person_kpts in enumerate(keypoints):
                if len(person_kpts) < 10:
                    continue

                hand_idx = 9 if self.hand_side == 'left' else 10
                nose_x, nose_y, nose_conf = person_kpts[0]
                hand_x, hand_y, hand_conf = person_kpts[hand_idx]
                
                if hand_conf > self.conf_threshold:
                    final_x, final_y = hand_x, hand_y
                    body_part = "Hand"
                else:
                    final_x, final_y = nose_x, nose_y
                    body_part = "Nose"
                
                detected_people.append({
                    "id": i,
                    "pos_x": float(final_x),
                    "pos_y": float(final_y),
                    "part_used": body_part,
                    "hand_confidence": float(hand_conf),
                    "nose_confidence": float(nose_conf)
                })
                
                if hand_conf > self.conf_threshold:
                    cv2.circle(annotated_frame, (int(hand_x), int(hand_y)), 10, (0, 0, 255), -1)
                else:
                    cv2.circle(annotated_frame, (int(nose_x), int(nose_y)), 10, (255, 0, 0), -1)
        
        pose_msg = String()
        pose_msg.data = json.dumps(detected_people)
        self.pose_pub.publish(pose_msg)
        
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
        annotated_msg.header.stamp = msg.header.stamp
        annotated_msg.header.frame_id = msg.header.frame_id
        self.annotated_image_pub.publish(annotated_msg)

def main():
    rclpy.init()
    node = PoseEstimationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()