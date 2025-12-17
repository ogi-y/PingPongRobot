from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import rclpy

class ImgPublisherNode(Node):
    def __init__(self):
        super().__init__('img_pub_node')
        self.declare_parameter('cam_source', '0')
        self.declare_parameter('fps', 30)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        
        cam_source = self.get_parameter('cam_source').get_parameter_value().string_value
        fps = self.get_parameter('fps').get_parameter_value().integer_value
        self.width = self.get_parameter('width').get_parameter_value().integer_value
        self.height = self.get_parameter('height').get_parameter_value().integer_value

        
        self.source = int(cam_source) if cam_source.isdigit() else cam_source
        self.publisher = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        if not self.cap.isOpened():
            self.get_logger().error("Could not open camera!")
            raise RuntimeError("Camera initialization failed")
        
        self.get_logger().info(f"Camera opened: {self.source}")
        self.timer = self.create_timer(1.0 / fps, self.timer_callback)
    
    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            if frame.shape[0] != self.height or frame.shape[1] != self.width:
                frame = cv2.resize(frame, (self.width, self.height))
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera_frame'
            self.publisher.publish(msg)
        else:
            self.get_logger().warn("Failed to read frame")
    
    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main():
    rclpy.init()
    node = ImgPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()