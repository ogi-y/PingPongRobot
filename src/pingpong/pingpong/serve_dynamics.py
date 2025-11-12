import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import math

class ServeCalculator(Node):
    def __init__(self):
        super().__init__('serve_calculator')
        self.subscription = self.create_subscription(
            String,
            '/serve_target',
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(String, '/serve_params', 10)
        self.get_logger().info('ServeCalculatorノード起動中...')

    def listener_callback(self, msg):
        try:
            serve_param = json.loads(msg.data)
            self.get_logger().info(f"受信: {serve_param}")

            # ----- 計算例 -----
            speed = serve_param["speed"]
            spin = serve_param["spin"]
            spin_dir = serve_param["spin_dir"]
            course_x = serve_param["course_x"]
            course_y = serve_param["course_y"]
            serve_pos = serve_param["serve_pos"]

            # 簡単な例：発射角度と速度ベクトルを計算
            # （ここは後であなたの物理モデルに置き換え）
            launch_angle = math.atan2(course_y - serve_pos, 1.37)  # 台の半分くらいの距離で仮計算
            vx = speed * math.cos(launch_angle)
            vy = speed * math.sin(launch_angle)
            omega = spin * (1 if spin_dir == "left" else -1)

            serve_cmd = {
                "vx": round(vx, 3),
                "vy": round(vy, 3),
                "omega": round(omega, 3),
                "angle_deg": round(math.degrees(launch_angle), 2)
            }

            msg_out = String()
            msg_out.data = json.dumps(serve_cmd)
            self.publisher.publish(msg_out)
            self.get_logger().info(f"計算結果送信: {serve_cmd}")

        except Exception as e:
            self.get_logger().error(f"エラー: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ServeCalculator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()