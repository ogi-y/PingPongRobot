import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import json


class SkeletonRecognitionNode(Node):
    def __init__(self):
        super().__init__('skeleton_recognition')
        self.sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.pub = self.create_publisher(String, 'skeleton', 10)
        self.bridge = CvBridge()
        
        # MediaPipeの初期化
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        self.get_logger().info('Skeleton recognition node started')

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # BGR to RGB
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # 骨格検出
        results = self.pose.process(image)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # 主要なランドマークの情報を抽出
            skeleton_data = {
                'detected': True,
                'landmarks': {}
            }
            
            # 主要なランドマーク（鼻、肩、肘、手首、腰、膝、足首）
            key_landmarks = [
                ('nose', self.mp_pose.PoseLandmark.NOSE),
                ('left_shoulder', self.mp_pose.PoseLandmark.LEFT_SHOULDER),
                ('right_shoulder', self.mp_pose.PoseLandmark.RIGHT_SHOULDER),
                ('left_elbow', self.mp_pose.PoseLandmark.LEFT_ELBOW),
                ('right_elbow', self.mp_pose.PoseLandmark.RIGHT_ELBOW),
                ('left_wrist', self.mp_pose.PoseLandmark.LEFT_WRIST),
                ('right_wrist', self.mp_pose.PoseLandmark.RIGHT_WRIST),
                ('left_hip', self.mp_pose.PoseLandmark.LEFT_HIP),
                ('right_hip', self.mp_pose.PoseLandmark.RIGHT_HIP),
                ('left_knee', self.mp_pose.PoseLandmark.LEFT_KNEE),
                ('right_knee', self.mp_pose.PoseLandmark.RIGHT_KNEE),
                ('left_ankle', self.mp_pose.PoseLandmark.LEFT_ANKLE),
                ('right_ankle', self.mp_pose.PoseLandmark.RIGHT_ANKLE),
            ]
            
            for name, landmark_id in key_landmarks:
                lm = landmarks[landmark_id]
                skeleton_data['landmarks'][name] = {
                    'x': float(lm.x),
                    'y': float(lm.y),
                    'z': float(lm.z),
                    'visibility': float(lm.visibility)
                }
            
            # JSON文字列として出力
            msg_str = json.dumps(skeleton_data)
            self.get_logger().info(f"Skeleton detected: {len(skeleton_data['landmarks'])} key landmarks")
            self.pub.publish(String(data=msg_str))
        else:
            skeleton_data = {'detected': False}
            msg_str = json.dumps(skeleton_data)
            self.pub.publish(String(data=msg_str))
            self.get_logger().info("No skeleton detected")

    def destroy_node(self):
        self.pose.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = SkeletonRecognitionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
