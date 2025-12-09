import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class IpWebcamPublisher(Node):
    def __init__(self):
        super().__init__('ip_webcam_publisher')
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.publisher_ = self.create_publisher(Image, 'image_raw', qos_profile)
        self.timer = self.create_timer(0.1, self.timer_callback) # 10Hz
        self.bridge = CvBridge()
        
        # 【重要】スマホのIPに合わせて書き換えてください
        # IP Webcamアプリの場合、末尾に /video をつけるとMJPEGストリームになります
        self.cap = cv2.VideoCapture('http://10.97.153.0:8080/video')
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            self.get_logger().error("カメラに接続できませんでした。IPアドレスを確認してください。")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            # OpenCV(BGR)をROSメッセージに変換してPublish
            frame = cv2.resize(frame, (640, 480))
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.publisher_.publish(msg)
            # self.get_logger().info('Publishing video frame')
        else:
            self.get_logger().warning('フレームの取得に失敗しました')

def main(args=None):
    rclpy.init(args=args)
    node = IpWebcamPublisher()
    rclpy.spin(node)
    node.cap.release()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()