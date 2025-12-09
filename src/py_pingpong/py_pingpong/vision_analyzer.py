import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import threading
import cv2
import json
from ultralytics import YOLO
from deepface import DeepFace

class VisionAnalyzer(Node):
    def __init__(self):
        super().__init__('vision_analyzer')
        
        # --- 設定 ---
        self.declare_parameter('source', '0') 
        source_param = self.get_parameter('source').get_parameter_value().string_value
        if source_param.isdigit():
            self.source = int(source_param)
        else:
            self.source = source_param # "http" -> そのまま
        self.source = "http://192.168.0.8:8080/video"  # 固定カメラソース設定

        self.age_interval = 30
        
        self.latest_frame = None
        self.lock = threading.Lock()
        
        # --- マルチスレッド設定 ---
        # 重い処理と軽い処理を並列に走らせるためのグループ設定
        self.callback_group = ReentrantCallbackGroup()

        self.pub_image = self.create_publisher(Image, '/camera/image_raw', 10)
        self.pub_result = self.create_publisher(String, '/vision/analysis', 10)
        self.bridge = CvBridge()

        self.get_logger().info("Loading YOLOv8 Pose model...")
        self.pose_model = YOLO('yolov8n-pose.pt') 
        #self.pose_model.to('cpu')
        self.get_logger().info(f"Opening camera source: {self.source}")
        self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            self.get_logger().error("Could not open camera!")
            exit()

        self.frame_count = 0
        self.current_age = "Estimating..."

        # 重い処理用のタイマー (callback_groupを指定して並列化許可)
        self.timer_age = self.create_timer(
            1.0, 
            self.process_age_estimation, 
            callback_group=self.callback_group
        )
        
        # メインループ用のタイマー (これもcallback_groupを指定)
        self.timer = self.create_timer(
            0.033, 
            self.process_frame, 
            callback_group=self.callback_group
        )

    def process_frame(self):
        ret, frame_raw = self.cap.read()
        if not ret:
            return
        frame = cv2.resize(frame_raw, (640, 480))
        # スレッドセーフに画像をコピー
        with self.lock:
            self.latest_frame = frame.copy()

        # 姿勢推定 (YOLOは高速なのでここでOK)
        results = self.pose_model(frame, verbose=False)
        annotated_frame = results[0].plot()
        
        detected_people = []
        if results[0].keypoints is not None:
            if results[0].keypoints.data.shape[1] > 0: # キーポイントが存在するか確認
                keypoints = results[0].keypoints.data.cpu().numpy()
                for i, person_kpts in enumerate(keypoints):
                    if len(person_kpts) > 0:
                        nose_x = person_kpts[0][0]
                        nose_y = person_kpts[0][1]
                        conf = person_kpts[0][2]
                        
                        person_info = {
                            "id": i,
                            "pos_x": float(nose_x),
                            "pos_y": float(nose_y),
                            "pose_detected": True if conf > 0.5 else False
                        }
                        detected_people.append(person_info)
        
        self.frame_count += 1

        # 描画
        cv2.putText(annotated_frame, f"Age: {self.current_age}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # データのPublish
        try:
            ros_image = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
            self.pub_image.publish(ros_image)
        except Exception as e:
            self.get_logger().error(f"Image publish failed: {e}")

        analysis_data = {
            "age": self.current_age,
            "people_count": len(detected_people),
            "people": detected_people
        }
        msg = String()
        msg.data = json.dumps(analysis_data)
        self.pub_result.publish(msg)

    def process_age_estimation(self):
        # 映像処理用のフレームを取得
        frame_to_process = None
        with self.lock:
            if self.latest_frame is not None:
                frame_to_process = self.latest_frame.copy()
        
        if frame_to_process is not None:
            try:
                analysis = DeepFace.analyze(frame_to_process, actions=['age'], enforce_detection=False, detector_backend='opencv')
                if isinstance(analysis, list) and len(analysis) > 0:
                    self.current_age = str(analysis[0]['age'])
                elif isinstance(analysis, dict):
                    self.current_age = str(analysis['age'])
                else:
                    # 顔が見つからなかった場合など
                    # self.get_logger().info("No face detected or analysis failed.")
                    pass
                    
            except Exception as e:
                # self.get_logger().warn(f"Age estimation failed: {e}")
                pass

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

def main(args=None):
    rclpy.init(args=args)
    node = VisionAnalyzer()
    
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()