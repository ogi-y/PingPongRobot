#!/usr/bin/env python3

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
        
        # パラメータ設定
        self.declare_parameter('detection_method', 'dnn')  # haar, dnn, or dlib
        self.declare_parameter('use_age_model', True)
        self.declare_parameter('min_face_size', 50)
        self.declare_parameter('confidence_threshold', 0.5)
        
        self.detection_method = self.get_parameter('detection_method').value
        self.use_age_model = self.get_parameter('use_age_model').value
        self.min_face_size = self.get_parameter('min_face_size').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        
        # CV Bridge初期化
        self.bridge = CvBridge()
        
        # 顔検出器の初期化
        self.init_face_detector()
        
        # 年齢推定モデルの初期化
        self.age_net = None
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
                         '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        
        if self.use_age_model:
            self.load_age_gender_model()
        
        # Subscriber: 画像トピック
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        # Publisher: 推定結果
        self.age_pub = self.create_publisher(Int32, '/estimated_age', 10)
        self.age_range_pub = self.create_publisher(String, '/estimated_age_range', 10)
        self.confidence_pub = self.create_publisher(Float32, '/age_confidence', 10)
        
        self.get_logger().info(f'Face Age Estimator initialized with {self.detection_method} detector')
    
    def init_face_detector(self):
        """顔検出器の初期化"""
        if self.detection_method == 'dnn':
            # DNNベースの顔検出（高精度）
            prototxt = 'deploy.prototxt'
            caffemodel = 'res10_300x300_ssd_iter_140000.caffemodel'
            
            if os.path.exists(prototxt) and os.path.exists(caffemodel):
                self.face_net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
                self.get_logger().info('Loaded DNN face detector')
            else:
                self.get_logger().warn('DNN model files not found, falling back to Haar Cascade')
                self.get_logger().warn('Download from: https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector')
                self.detection_method = 'haar'
                self.init_haar_cascade()
        
        elif self.detection_method == 'dlib':
            try:
                import dlib
                self.dlib = dlib
                self.face_detector = dlib.get_frontal_face_detector()
                self.get_logger().info('Loaded dlib face detector')
            except ImportError:
                self.get_logger().warn('dlib not installed, falling back to Haar Cascade')
                self.get_logger().warn('Install with: sudo apt install python3-dlib')
                self.detection_method = 'haar'
                self.init_haar_cascade()
        
        else:  # haar
            self.init_haar_cascade()
    
    def init_haar_cascade(self):
        """Haar Cascade初期化"""
        cascade_path = self.find_cascade_file()
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise Exception('Failed to load Haar Cascade')
        self.get_logger().info('Loaded Haar Cascade face detector')
    
    def find_cascade_file(self):
        """Haar Cascadeファイルのパスを検索"""
        possible_paths = [
            '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        ]
        
        try:
            cv2_data_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cv2_data_path):
                return cv2_data_path
        except AttributeError:
            pass
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError('haarcascade_frontalface_default.xml not found')
    
    def load_age_gender_model(self):
        """年齢・性別推定モデルの読み込み"""
        age_proto = 'age_deploy.prototxt'
        age_model = 'age_net.caffemodel'
        
        if os.path.exists(age_proto) and os.path.exists(age_model):
            try:
                self.age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
                # モデルが正しく読み込まれたか確認
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
    
    def detect_faces_dnn(self, image):
        """DNNベースの顔検出"""
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), 
                                     (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype(int)
                faces.append((x1, y1, x2-x1, y2-y1, confidence))
        
        return faces
    
    def detect_faces_dlib(self, image):
        """dlibベースの顔検出"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray, 1)
        
        result = []
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            result.append((x, y, w, h, 1.0))
        
        return result
    
    def detect_faces_haar(self, image):
        """Haar Cascadeベースの顔検出"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, 
            minSize=(self.min_face_size, self.min_face_size)
        )
        
        return [(x, y, w, h, 1.0) for (x, y, w, h) in faces]
    
    def estimate_age_dnn(self, face_img):
        """DNNモデルを使用した年齢推定（連続値）"""
        try:
            # 顔画像のサイズチェック
            if face_img.shape[0] < 20 or face_img.shape[1] < 20:
                self.get_logger().warn('Face image too small for age estimation')
                return None, None, 0.0
            
            # モデルが正しく読み込まれているか確認
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
            
            # 各年齢範囲の確率
            probabilities = age_preds[0]
            
            # 方法1: 最も確率の高い範囲の中央値（従来の方法）
            age_idx = probabilities.argmax()
            age_range = self.age_list[age_idx]
            confidence = float(probabilities[age_idx])
            
            # 方法2: 確率分布から期待値を計算（連続値推定）
            age_ranges_numeric = [1, 5, 10, 17, 28, 40, 50, 70]
            estimated_age_continuous = sum(age * prob for age, prob in zip(age_ranges_numeric, probabilities))
            
            # 方法3: 加重平均（上位3つの確率を使用）
            top_indices = np.argsort(probabilities)[-3:][::-1]
            weighted_age = sum(age_ranges_numeric[i] * probabilities[i] for i in top_indices)
            weighted_sum = sum(probabilities[i] for i in top_indices)
            if weighted_sum > 0:
                estimated_age_refined = weighted_age / weighted_sum
            else:
                estimated_age_refined = estimated_age_continuous
            
            # より自然な推定値を使用
            estimated_age = int(round(estimated_age_refined))
            
            return estimated_age, age_range, confidence
        except Exception as e:
            self.get_logger().error(f'Age estimation failed: {e}')
            return None, None, 0.0
    
    def estimate_age_heuristic(self, face_img, face_size):
        """ヒューリスティックな年齢推定（モデルなし）"""
        # 顔のサイズと画像特徴から簡易推定
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # しわや質感の分析（簡易版）
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 顔サイズが大きいほど近くにいる（子供の可能性）
        # テクスチャが複雑なほど年齢が高い傾向
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
            # ROS ImageメッセージをOpenCV形式に変換
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # 顔検出
            if self.detection_method == 'dnn':
                faces = self.detect_faces_dnn(cv_image)
            elif self.detection_method == 'dlib':
                faces = self.detect_faces_dlib(cv_image)
            else:  # haar
                faces = self.detect_faces_haar(cv_image)
            
            if len(faces) == 0:
                self.get_logger().debug('No face detected')
                return
            
            # 最大の顔を選択
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            (x, y, w, h, conf) = faces[0]
            
            # 顔領域の抽出
            x, y = max(0, x), max(0, y)
            face_img = cv_image[y:y+h, x:x+w]
            
            # 年齢推定
            if self.age_net is not None:
                age, age_range, confidence = self.estimate_age_dnn(face_img)
                # DNNでの推定が失敗した場合はヒューリスティックにフォールバック
                if age is None:
                    age, age_range, confidence = self.estimate_age_heuristic(face_img, w*h)
            else:
                age, age_range, confidence = self.estimate_age_heuristic(face_img, w*h)
            
            # トピックに配信
            age_msg = Int32()
            age_msg.data = age
            self.age_pub.publish(age_msg)
            
            age_range_msg = String()
            age_range_msg.data = age_range
            self.age_range_pub.publish(age_range_msg)
            
            confidence_msg = Float32()
            confidence_msg.data = confidence
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