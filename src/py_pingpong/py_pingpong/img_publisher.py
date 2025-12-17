from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import rclpy
import os

class ImgPublisherNode(Node):
    def __init__(self):
        super().__init__('img_pub_node')
        
        # パラメータ宣言
        self.declare_parameter('cam_source', '0')
        self.declare_parameter('fps', 30)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('loop_video', True)  # 動画ファイルのみ有効
        self.declare_parameter('use_v4l2', True)   # /dev/videoデバイス用
        
        cam_source = self.get_parameter('cam_source').get_parameter_value().string_value
        fps = self.get_parameter('fps').get_parameter_value().integer_value
        self.width = self.get_parameter('width').get_parameter_value().integer_value
        self. height = self.get_parameter('height').get_parameter_value().integer_value
        self.loop_video = self.get_parameter('loop_video').get_parameter_value().bool_value
        use_v4l2 = self.get_parameter('use_v4l2').get_parameter_value().bool_value
        
        # ソース種別の判定
        self.source_type = self._detect_source_type(cam_source)
        
        self.publisher = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()
        
        # ソース種別に応じてカメラ/動画を開く
        if self.source_type == 'CAMERA_INDEX':
            # 数字指定：カメラインデックス
            self.source = int(cam_source)
            self.is_video_file = False
            self.get_logger().info(f"Opening camera by index: {self.source}")
            
            if use_v4l2:
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
                self.get_logger().info("Using V4L2 backend")
            else:
                self. cap = cv2.VideoCapture(self.source)
                self.get_logger().info("Using default backend")
            
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
        elif self.source_type == 'DEVICE_PATH':
            # /dev/video* デバイスパス
            self.source = cam_source
            self.is_video_file = False
            self.get_logger().info(f"Opening camera device: {self.source}")
            
            if use_v4l2:
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
                self.get_logger().info("Using V4L2 backend")
            else:
                self. cap = cv2.VideoCapture(self.source)
                self.get_logger().info("Using default backend")
            
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
        elif self.source_type == 'VIDEO_FILE':
            # 動画ファイル
            self.source = cam_source
            self. is_video_file = True
            
            # ファイルの存在確認
            if not os.path.exists(self.source):
                self.get_logger().error(f"Video file not found: {self.source}")
                raise FileNotFoundError(f"Video file not found: {self.source}")
            
            self.get_logger().info(f"Opening video file: {self. source}")
            self.cap = cv2.VideoCapture(self.source)
            
        else:
            self.get_logger().error(f"Unknown source type: {cam_source}")
            raise ValueError(f"Unknown source type: {cam_source}")
        
        # カメラ/動画が開けたか確認
        if not self.cap.isOpened():
            self.get_logger().error(f"Could not open source: {cam_source}")
            raise RuntimeError("Video/Camera initialization failed")
        
        # 動画ファイルの場合、詳細情報を表示
        if self.is_video_file:
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / video_fps if video_fps > 0 else 0
            
            self. get_logger().info(
                f"Video info: {video_width}x{video_height}, "
                f"{total_frames} frames, {video_fps:.1f} FPS, {duration:.1f}s"
            )
            if self.loop_video:
                self.get_logger().info("Loop mode:  ENABLED")
            else:
                self.get_logger().info("Loop mode: DISABLED (will stop at end)")
        else:
            # カメラの場合、設定を試みる
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.get_logger().info(
                f"Camera configured: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS"
            )
        
        self.frame_count = 0
        self.get_logger().info(f"Publishing at {fps} FPS to /camera/image_raw")
        self.timer = self.create_timer(1.0 / fps, self.timer_callback)
    
    def _detect_source_type(self, source_str):
        """
        ソースの種別を判定
        - 数字のみ → CAMERA_INDEX
        - /dev/video* → DEVICE_PATH
        - その他 → VIDEO_FILE
        """
        # 数字のみかチェック
        if source_str.isdigit():
            return 'CAMERA_INDEX'
        
        # /dev/video で始まるかチェック
        if source_str.startswith('/dev/video'):
            return 'DEVICE_PATH'
        
        # それ以外は動画ファイルとみなす
        return 'VIDEO_FILE'
    
    def timer_callback(self):
        ret, frame = self.cap. read()
        
        # フレーム読み込み失敗時の処理
        if not ret: 
            if self.is_video_file and self.loop_video:
                # 動画ファイルのループ再生
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                self.get_logger().info("Video looped to start")
                
                if not ret:
                    self.get_logger().error("Failed to loop video")
                    return
            else:
                if self.is_video_file:
                    self.get_logger().info("Video ended.  Stopping publisher...")
                else:
                    self.get_logger().warn("Failed to read frame from camera")
                return
        
        # フレームをリサイズ
        if frame.shape[1] != self.width or frame. shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        
        # ROS2メッセージに変換
        try:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera_frame'
            self.publisher.publish(msg)
            self.frame_count += 1
        except Exception as e:
            self.get_logger().error(f"Failed to publish frame: {e}")
            return
        
        # 定期的なログ出力（100フレームごと）
        if self.frame_count % 100 == 0:
            if self.is_video_file:
                current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.get_logger().info(
                    f"Published frame {current_frame}/{total_frames} "
                    f"(Total: {self.frame_count} frames)"
                )
            else:
                self.get_logger().debug(f"Published {self.frame_count} frames from camera")
    
    def destroy_node(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.get_logger().info("Camera/Video source released")
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

if __name__ == '__main__': 
    main()