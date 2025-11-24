import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

import cv2
import json
from ultralytics import YOLO
from deepface import DeepFace

class VisionAnalyzer(Node):
    def __init__(self):
        super().__init__('vision_analyzer')
        
        # --- 設定 ---
        self.camera_id = 0       # USBカメラのID (0 or 2 etc.)
        self.age_interval = 30   # 年齢推定を行う間隔 (フレーム数)
        
        # --- ROS通信設定 ---
        self.pub_image = self.create_publisher(Image, '/camera/image_raw', 10)
        self.pub_result = self.create_publisher(String, '/vision/analysis', 10)
        self.bridge = CvBridge()

        # --- AIモデルのロード ---
        self.get_logger().info("Loading YOLOv8 Pose model...")
        self.pose_model = YOLO('yolov8n-pose.pt') # 初回はダウンロード走ります
        
        self.get_logger().info("Vision Analyzer Started.")

        # --- カメラ起動 ---
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            self.get_logger().error("Could not open camera!")
            exit()

        # --- ループ用変数 ---
        self.frame_count = 0
        self.current_age = "Estimating..."
        self.timer = self.create_timer(0.033, self.process_frame) # 約30fps

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # 1. 姿勢推定 (YOLOv8) - 毎フレーム実行
        results = self.pose_model(frame, verbose=False)
        annotated_frame = results[0].plot() # 骨格を描画

        # 検出された人の情報リスト
        detected_people = []

        # キーポイント情報の抽出
        if results[0].keypoints is not None:
            # dataは (人数, 17個の関節, [x, y, conf]) の形
            keypoints = results[0].keypoints.data.cpu().numpy()
            
            for i, person_kpts in enumerate(keypoints):
                # 鼻(0番)の座標を取得 (簡易的な位置特定のため)
                nose_x = person_kpts[0][0]
                nose_y = person_kpts[0][1]
                conf = person_kpts[0][2]
                
                person_info = {
                    "id": i,
                    "pos_x": float(nose_x), # 画像内のX座標
                    "pos_y": float(nose_y),
                    "pose_detected": True if conf > 0.5 else False
                }
                detected_people.append(person_info)

        # 2. 年齢推定 (DeepFace) - 重いので間引く
        if self.frame_count % self.age_interval == 0:
            try:
                # DeepFaceは顔検出も含んでいる
                # enforce_detection=Falseにすると顔がなくてもエラーにならない
                analysis = DeepFace.analyze(frame, actions=['age'], enforce_detection=False, silent=True)
                if isinstance(analysis, list) and len(analysis) > 0:
                    self.current_age = str(analysis[0]['age'])
            except Exception as e:
                pass # 顔が見つからない場合などは無視
        
        self.frame_count += 1

        # 3. 画面への情報の描画
        cv2.putText(annotated_frame, f"Age: {self.current_age}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 4. データのPublish
        # 画像データの送信
        try:
            ros_image = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
            self.pub_image.publish(ros_image)
        except Exception as e:
            self.get_logger().error(f"Image publish failed: {e}")

        # 分析結果(JSON)の送信
        analysis_data = {
            "age": self.current_age,
            "people_count": len(detected_people),
            "people": detected_people
        }
        msg = String()
        msg.data = json.dumps(analysis_data)
        self.pub_result.publish(msg)

        # デバッグ画面表示 (ROS経由で見ない場合用、不要ならコメントアウト)
        cv2.imshow("Vision Analyzer", annotated_frame)
        cv2.waitKey(1)

    def __del__(self):
        self.cap.release()

def main(args=None):
    rclpy.init(args=args)
    node = VisionAnalyzer()
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