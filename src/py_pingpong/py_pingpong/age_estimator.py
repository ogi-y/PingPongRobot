import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import json
from deepface import DeepFace

class AgeEstimationNode(Node):
    def __init__(self):
        super().__init__('age_estimation_node')
        
        self.declare_parameter('detector_backend', 'opencv')
        self.declare_parameter('process_interval', 120)
        self.declare_parameter('enforce_detection', False)
        
        self.detector_backend = self.get_parameter('detector_backend').get_parameter_value().string_value
        self.process_interval = self.get_parameter('process_interval').get_parameter_value().integer_value
        self.enforce_detection = self.get_parameter('enforce_detection').get_parameter_value().bool_value
        
        self.bridge = CvBridge()
        self.frame_count = 0
        
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        self.age_pub = self.create_publisher(String, '/age/estimation_result', 10)
        self.annotated_image_pub = self.create_publisher(Image, '/age/annotated_image', 10)
        
        self.get_logger().info(f'Age estimation node started. Detector: {self.detector_backend}')
    
    def image_callback(self, msg):
        self.frame_count += 1
        
        if self.frame_count % self.process_interval != 0:
            return
        
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        annotated_frame = frame.copy()
        detected_faces = []
        
        try:
            results = DeepFace.analyze(
                frame,
                actions=['age'],
                detector_backend=self.detector_backend,
                enforce_detection=self.enforce_detection,
                silent=True
            )
            
            if not isinstance(results, list):
                results = [results]
            
            for i, face in enumerate(results):
                age = face.get('age', 0)
                gender = face.get('dominant_gender', 0)
                region = face.get('region', {})
                
                x = region.get('x', 0)
                y = region.get('y', 0)
                w = region.get('w', 0)
                h = region.get('h', 0)
                
                detected_faces.append({
                    "id": i,
                    "age": int(age),
                    "gender": gender,
                    "bbox": {"x": x, "y": y, "width": w, "height": h}
                })
                
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                label = f"Age: {int(age)}, {gender}"
                cv2.putText(annotated_frame, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            self.get_logger().info(f'Detected {len(detected_faces)} face(s)')
            
        except Exception as e:
            self.get_logger().warn(f'Face detection failed: {str(e)}')
            detected_faces = []
        
        age_msg = String()
        age_msg.data = json.dumps(detected_faces)
        self.age_pub.publish(age_msg)
        
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
        annotated_msg.header.stamp = msg.header.stamp
        annotated_msg.header.frame_id = msg.header.frame_id
        self.annotated_image_pub.publish(annotated_msg)

def main():
    rclpy.init()
    node = AgeEstimationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()