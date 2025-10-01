import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np

# カスタムメッセージタイプを定義
from std_msgs.msg import String
import json

class FaceInfo:
    def __init__(self, face_id, bbox, face_image):
        self.face_id = face_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.face_image = face_image

class PersonDetectorNode(Node):
    def __init__(self):
        super().__init__('person_detector_node')
        
        # サブスクリプション
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10)
        
        # パブリッシャー - 検出された顔情報と顔画像を送信
        self.face_info_publisher = self.create_publisher(
            String,  # 顔情報（JSON形式）
            '/face_info',
            10)
        
        self.face_image_publisher = self.create_publisher(
            Image,   # 顔画像
            '/face_images',
            10)
        
        # ブリッジとモデル
        self.bridge = CvBridge()
        self.person_model = YOLO('yolov8n.pt')  # 人物検出用
        
        # 顔検出用のカスケード分類器
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # パラメータ
        self.declare_parameter('min_face_size', 30)
        self.declare_parameter('face_detection_scale', 1.1)
        self.declare_parameter('min_neighbors', 5)
        self.declare_parameter('face_margin', 10)  # 顔周りのマージン
        
        self.face_id_counter = 0
        
        self.get_logger().info('Person Detector with Face Detection initialized')

    def detect_faces_in_person_bbox(self, image, person_bbox):
        """人物のバウンディングボックス内で顔を検出"""
        x1, y1, x2, y2 = person_bbox
        
        # 人物領域を切り出し
        person_roi = image[y1:y2, x1:x2]
        
        if person_roi.size == 0:
            return []
        
        # グレースケール変換
        gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
        
        # パラメータ取得
        min_face_size = self.get_parameter('min_face_size').value
        scale_factor = self.get_parameter('face_detection_scale').value
        min_neighbors = self.get_parameter('min_neighbors').value
        
        # 顔検出
        faces = self.face_cascade.detectMultiScale(
            gray_roi,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(min_face_size, min_face_size)
        )
        
        detected_faces = []
        for (fx, fy, fw, fh) in faces:
            # グローバル座標に変換
            global_x1 = x1 + fx
            global_y1 = y1 + fy
            global_x2 = global_x1 + fw
            global_y2 = global_y1 + fh
            
            # マージンを追加して顔画像を切り出し
            margin = self.get_parameter('face_margin').value
            crop_x1 = max(0, global_x1 - margin)
            crop_y1 = max(0, global_y1 - margin)
            crop_x2 = min(image.shape[1], global_x2 + margin)
            crop_y2 = min(image.shape[0], global_y2 + margin)
            
            face_image = image[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if face_image.size > 0:
                face_info = FaceInfo(
                    face_id=self.face_id_counter,
                    bbox=[global_x1, global_y1, global_x2, global_y2],
                    face_image=face_image
                )
                detected_faces.append(face_info)
                self.face_id_counter += 1
        
        return detected_faces

    def publish_face_data(self, faces, timestamp):
        """検出された顔情報を複数のトピックでパブリッシュ"""
        if not faces:
            return
        
        # 顔情報をJSONでパブリッシュ
        face_info_data = {
            'timestamp': int(timestamp.sec * 10**9 + timestamp.nanosec),
            'faces': []
        }
        
        for face in faces:
            face_data = {
                'face_id': int(face.face_id),
                'bbox': [int(x) for x in face.bbox],
                'width': int(face.face_image.shape[1]),
                'height': int(face.face_image.shape[0])
            }
            face_info_data['faces'].append(face_data)
            
            # 個別の顔画像をパブリッシュ
            try:
                face_msg = self.bridge.cv2_to_imgmsg(face.face_image, encoding='bgr8')
                face_msg.header.stamp = timestamp
                face_msg.header.frame_id = f'face_{face.face_id}'
                self.face_image_publisher.publish(face_msg)
            except Exception as e:
                self.get_logger().error(f'Error publishing face image {face.face_id}: {str(e)}')
        
        # 顔情報JSONをパブリッシュ
        info_msg = String()
        info_msg.data = json.dumps(face_info_data)
        self.face_info_publisher.publish(info_msg)
        
        self.get_logger().info(f'Published {len(faces)} faces for age estimation')

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # 人物検出
            person_results = self.person_model(cv_image)
            person_boxes = person_results[0].boxes
            
            all_faces = []
            
            if person_boxes is not None:
                for box in person_boxes:
                    cls_id = int(box.cls[0])
                    if cls_id == 0:  # 0は"person"クラス
                        confidence = float(box.conf[0])
                        if confidence > 0.5:  # 信頼度しきい値
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # 人物バウンディングボックスを描画
                            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(cv_image, f'Person ({confidence:.2f})', 
                                      (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                            
                            # この人物領域内で顔検出
                            faces = self.detect_faces_in_person_bbox(cv_image, [x1, y1, x2, y2])
                            all_faces.extend(faces)
                            
                            # 検出された顔を描画
                            for face in faces:
                                fx1, fy1, fx2, fy2 = face.bbox
                                cv2.rectangle(cv_image, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
                                cv2.putText(cv_image, f'Face ID: {face.face_id}', 
                                          (fx1, fy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 顔データをパブリッシュ
            if all_faces:
                self.publish_face_data(all_faces, msg.header.stamp)
            
            # 結果を表示
            cv2.imshow("Person and Face Detection", cv_image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f'Error in image processing: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = PersonDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()