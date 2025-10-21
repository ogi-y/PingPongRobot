# ...existing code...
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, String, Float32
from cv_bridge import CvBridge
import cv2
import numpy as np
import os

class FaceAgeEstimatorNode(Node):
    def __init__(self):
        super().__init__('face_age_estimator')
        
        # パラメータ設定（モデル選択は廃止、DNN のみ）
        self.declare_parameter('use_age_model', True)
        self.declare_parameter('min_face_size', 50)
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('face_proto', 'deploy.prototxt')
        self.declare_parameter('face_model', 'res10_300x300_ssd_iter_140000.caffemodel')
        self.declare_parameter('age_proto', 'age_deploy.prototxt')
        self.declare_parameter('age_model', 'age_net.caffemodel')
        
        self.use_age_model = self.get_parameter('use_age_model').value
        self.min_face_size = self.get_parameter('min_face_size').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.image_topic = self.get_parameter('image_topic').value
        self.face_proto = self.get_parameter('face_proto').value
        self.face_model = self.get_parameter('face_model').value
        self.age_proto = self.get_parameter('age_proto').value
        self.age_model = self.get_parameter('age_model').value
        
        # CV Bridge初期化
        self.bridge = CvBridge()
        
        # 顔検出器（DNNのみ）初期化
        self.face_net = None
        self.init_face_detector()
        
        # 年齢推定モデルの初期化
        self.age_net = None
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
                         '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        
        if self.use_age_model:
            self.load_age_gender_model(self.age_proto, self.age_model)
        
        # Subscriber: 画像トピック
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )
        
        # Publisher: 推定結果
        self.age_pub = self.create_publisher(Int32, '/estimated_age', 10)
        self.age_range_pub = self.create_publisher(String, '/estimated_age_range', 10)
        self.confidence_pub = self.create_publisher(Float32, '/age_confidence', 10)
        
        self.get_logger().info('Face Age Estimator initialized (DNN only)')
    
    def init_face_detector(self):
        """DNNベースの顔検出器を読み込む（唯一の検出器）"""
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
    
    def load_age_gender_model(self, age_proto='age_deploy.prototxt', age_model='age_net.caffemodel'):
        """年齢推定モデルの読み込み"""
        if os.path.exists(age_proto) and os.path.exists(age_model):
            try:
                self.age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
                if self.age_net.empty():
                    self.get_logger().warn('Age model is empty')
                    self.age_net = None
                else:
                    self.get_logger().info('Loaded age estimation model')
                return
            except Exception as e:
                self.get_logger().warn(f'Could not load age model: {e}')
                self.age_net = None
        
        self.get_logger().warn('Age model not found. Using heuristic age estimation.')
        self.get_logger().info('Download models from: https://github.com/GilLevi/AgeGenderDeepLearning')
    
    def detect_faces(self, image):
        """DNNベースの顔検出（単一関数）"""
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
                # 幅・高さが最小より小さければ除外
                wbox, hbox = x2 - x1, y2 - y1
                if wbox < self.min_face_size or hbox < self.min_face_size:
                    continue
                faces.append((x1, y1, wbox, hbox, confidence))
        
        return faces
    
    def estimate_age_dnn(self, face_img):
        """DNNモデルを使用した年齢推定（連続値）"""
        try:
            if face_img is None or face_img.size == 0:
                self.get_logger().warn('Empty face image for age estimation')
                return None, None, 0.0
            
            if self.age_net is None or self.age_net.empty():
                self.get_logger().warn('Age model not available')
                return None, None, 0.0
            
            blob = cv2.dnn.blobFromImage(
                face_img, 1.0, (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False
            )
            
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            probabilities = age_preds[0]
            age_idx = probabilities.argmax()
            age_range = self.age_list[age_idx]
            confidence = float(probabilities[age_idx])
            
            age_ranges_numeric = [1, 5, 10, 17, 28, 40, 50, 70]
            top_indices = np.argsort(probabilities)[-3:][::-1]
            weighted_age = sum(age_ranges_numeric[i] * probabilities[i] for i in top_indices)
            weighted_sum = sum(probabilities[i] for i in top_indices)
            if weighted_sum > 0:
                estimated_age_refined = weighted_age / weighted_sum
            else:
                estimated_age_refined = sum(age * prob for age, prob in zip(age_ranges_numeric, probabilities))
            
            estimated_age = int(round(estimated_age_refined))
            return estimated_age, age_range, confidence
        except Exception as e:
            self.get_logger().error(f'Age estimation failed: {e}')
            return None, None, 0.0
    
    def estimate_age_heuristic(self, face_img, face_size):
        """モデルがないときの簡易年齢推定（保険）"""
        if face_img is None or face_img.size == 0:
            return None, None, 0.0
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if face_size < 80:
            age = 25
            age_range = '(25-32)'
        elif laplacian_var < 300:
            age = 15
            age_range = '(15-20)'
        elif laplacian_var < 500:
            age = 28
            age_range = '(25-32)'
        elif laplacian_var < 700:
            age = 40
            age_range = '(38-43)'
        else:
            age = 50
            age_range = '(48-53)'
        return age, age_range, 0.6
    
    def image_callback(self, msg):
        """画像トピックのコールバック関数"""
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
            
            if self.age_net is not None:
                age, age_range, confidence = self.estimate_age_dnn(face_img)
                if age is None:
                    age, age_range, confidence = self.estimate_age_heuristic(face_img, w*h)
            else:
                age, age_range, confidence = self.estimate_age_heuristic(face_img, w*h)
            
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
# ...existing code...