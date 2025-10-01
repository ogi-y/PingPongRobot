import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class PersonDetectorNode(Node):
    def __init__(self):
        super().__init__('person_detector_node')
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',  # 画像トピック名
            self.image_callback,
            10)
        self.bridge = CvBridge()
        # YOLOv8の人物検出モデル（事前学習済みCOCOモデル）
        self.model = YOLO('yolov8n.pt')

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(cv_image)
        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:  # 0は"person"クラス
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(cv_image, 'person', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Person Detection", cv_image)
        cv2.waitKey(1)

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