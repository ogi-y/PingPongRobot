import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2

class ImageSizePublisher(Node):
    def __init__(self):
        super().__init__('image_size_publisher')
        self.sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.pub = self.create_publisher(String, 'pos_ball', 10)
        self.bridge = CvBridge()

    def ball_detection(self, cv_image):
        # ここにボール検出処理を書く
        return (pos2d, radius)  # 検出した座標を返す
    
    def estimate_ball_position(self, ball_2d_pos, ball_radius):
        # ここに3D位置推定処理を書く
        return (pred_x, pred_y, pred_z)  # 予測した3D位置を返す

    def predict_ball_trajectory(self, ball_2d_pos, ball_radius):
        # ここに軌道予測処理を書く
        return prediction  # 予測位置を返す
    
    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        ball_position_2d, ball_radius = self.ball_detection(cv_image)
        self.get_logger().info(f"Detected ball position: {ball_position_2d}, radius: {ball_radius}")
        predicted_position = self.estimate_ball_position(ball_position_2d, ball_radius)
        self.get_logger().info(f"Predicted ball position: {predicted_position}")
        trajectory = self.predict_ball_trajectory(ball_position_2d, ball_radius)
        self.get_logger().info(f"Predicted ball trajectory: {trajectory}")

        self.pub.publish(String(data=f"trajectory_data{trajectory}"))

def main():
    rclpy.init()
    node = ImageSizePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()