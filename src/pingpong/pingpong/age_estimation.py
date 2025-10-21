import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, String, Float32
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from deepface import DeepFace

class FaceAgeEstimatorNode(Node):
    def __init__(self):
        super().__init__('face_age_estimator')
        
        self.declare_parameter('min_face_size', 50)
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('face_proto', 'deploy.prototxt')
        self.declare_parameter('face_model', 'res10_300x300_ssd_iter_140000.caffemodel')
        
        self.min_face_size = self.get_parameter('min_face_size').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.image_topic = self.get_parameter('image_topic').value
        self.face_proto = self.get_parameter('face_proto').value
        self.face_model = self.get_parameter('face_model').value
        
        self.bridge = CvBridge()
        self.face_net = None
        self.init_face_detector()
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
                         '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )
        
        self.age_pub = self.create_publisher(Int32, '/estimated_age', 10)
        self.age_range_pub = self.create_publisher(String, '/estimated_age_range', 10)
        self.confidence_pub = self.create_publisher(Float32, '/age_confidence', 10)
        
        self.get_logger().info('Face Age Estimator initialized (DNN for detection, DeepFace for age)')
    
    def init_face_detector(self):
        if os.path.exists(self.face_proto) and os.path.exists(self.face_model):
            try:
                self.face_net = cv2.dnn.readNetFromCaffe(self.face_proto, self.face_model)
                if self.face_net.empty():
                    self.get_logger().error('Loaded face detector is empty')
                    self.face_net = None
                else:
                    self.get_logger().info('Loaded DNN face detector')
            except Exception as e:
                self.get_logger().error(f'Failed to load DNN face detector: {e}')
                self.face_net = None
        else:
            self.get_logger().error(
                f'Face detection model files not found: {self.face_proto}, {self.face_model}'
            )
            self.get_logger().info(
                'Get models from: https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector'
            )
            self.face_net = None
    
    def detect_faces(self, image):
        if self.face_net is None:
            return []
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), 
                                     (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype(int)
                wbox, hbox = x2 - x1, y2 - y1
                if wbox < self.min_face_size or hbox < self.min_face_size:
                    continue
                faces.append((x1, y1, wbox, hbox, confidence))
        
        return faces
    
    def estimate_age_deepface(self, face_img):
        try:
            if face_img is None or face_img.size == 0:
                self.get_logger().warn('Empty face image for age estimation')
                return None, None, 0.0
            
            result = DeepFace.analyze(face_img, actions=['age'], enforce_detection=False)
            if isinstance(result, list) and len(result) > 0:
                age = result[0]['age']
                if age <= 3:
                    age_range = '(0-2)'
                elif age <= 7:
                    age_range = '(4-6)'
                elif age <= 14:
                    age_range = '(8-12)'
                elif age <= 22:
                    age_range = '(15-20)'
                elif age <= 35:
                    age_range = '(25-32)'
                elif age <= 45:
                    age_range = '(38-43)'
                elif age <= 56:
                    age_range = '(48-53)'
                else:
                    age_range = '(60-100)'
                confidence = 0.8  # DeepFaceの信頼度は固定値として扱う
                return age, age_range, confidence
            else:
                self.get_logger().warn('DeepFace analysis failed')
                return None, None, 0.0
        except Exception as e:
            self.get_logger().error(f'Age estimation failed: {e}')
            return None, None, 0.0
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            faces = self.detect_faces(cv_image)
            if len(faces) == 0:
                self.get_logger().debug('No face detected')
                return
            
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            (x, y, w, h, conf) = faces[0]
            x, y = max(0, x), max(0, y)
            face_img = cv_image[y:y+h, x:x+w]
            
            age, age_range, confidence = self.estimate_age_deepface(face_img)
            
            age_msg = Int32()
            age_msg.data = int(age) if age is not None else -1
            self.age_pub.publish(age_msg)
            
            age_range_msg = String()
            age_range_msg.data = age_range if age_range is not None else ''
            self.age_range_pub.publish(age_range_msg)
            
            confidence_msg = Float32()
            confidence_msg.data = float(confidence) if confidence is not None else 0.0
            self.confidence_pub.publish(confidence_msg)
            
            self.get_logger().info(
                f'Age: {age} {age_range} (conf: {confidence:.2f}, face: {w}x{h})'
            )
                
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = FaceAgeEstimatorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()