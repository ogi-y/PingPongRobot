# 中身は適当
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import math

class MotorController(Node):
    def __init__(self):
        super().__init__('motor_controller')
        # serve_calculatorからのコマンド受信用
        self.subscription = self.create_subscription(
            String,
            '/serve_params',   # serve_calculatorの出力トピック名
            self.command_callback,
            10
        )

        # モーター制御用（ここでは例として文字列で送信）
        self.motor_pub = self.create_publisher(String, '/motor_commands', 10)
        self.get_logger().info("MotorControllerノード起動")

    def command_callback(self, msg):
        try:
            data = json.loads(msg.data)
            robo_pos = data.get("robo_pos", 0.0)
            roll = data.get("roll", 0.0)
            pitch = data.get("pitch", 0.0)
            yaw = data.get("yaw", 0.0)
            v = data.get("v", 0.0)

            # --- 各モーター指令を計算 ---
            commands = self.calculate_motor_commands(robo_pos, roll, pitch, yaw, v)

            # --- モータードライバへ送信 ---
            cmd_str = json.dumps(commands)
            self.motor_pub.publish(String(data=cmd_str))

            self.get_logger().info(f"モーター指令送信: {cmd_str}")

        except Exception as e:
            self.get_logger().error(f"コマンド解析エラー: {e}")

    def calculate_motor_commands(self, robo_pos, roll, pitch, yaw, v):
        """
        各モーターの指令値を計算して辞書で返す
        """
        # 位置モーター（ステージ移動など）
        motor_pos = self.robo_pos_to_motor_steps(robo_pos)

        # サーブ角制御（ロール・ピッチ・ヨー）
        motor_roll = self.angle_to_servo(roll)
        motor_pitch = self.angle_to_servo(pitch)
        motor_yaw = self.angle_to_servo(yaw)

        # 発射モーター（左右ホイール）
        left_rpm, right_rpm = self.v_to_wheel_rpm(v, roll)

        # ボール供給モーター（1回転分を仮定）
        feeder_cmd = 1.0

        return {
            "pos_motor": motor_pos,
            "roll_motor": motor_roll,
            "pitch_motor": motor_pitch,
            "yaw_motor": motor_yaw,
            "fire_left_rpm": left_rpm,
            "fire_right_rpm": right_rpm,
            "feeder_motor": feeder_cmd
        }

    def robo_pos_to_motor_steps(self, pos):
        """ロボ位置をモーターステップに変換（仮）"""
        return int(pos * 1000)  # 例：1.0m → 1000ステップ

    def angle_to_servo(self, angle_deg):
        """角度→サーボ角度(0〜180度想定)"""
        return int(max(0, min(180, angle_deg)))

    def v_to_wheel_rpm(self, v, roll_angle):
        """打ち出し速度v[m/s]から左右ホイールの回転数を計算"""
        wheel_radius = 0.03  # [m]
        base_rpm = (v / (2 * math.pi * wheel_radius)) * 60
        # 右スピンなどで回転差をつける（簡易モデル）
        diff = math.cos(math.radians(roll_angle)) * 50
        left_rpm = round(base_rpm + diff, 1)
        right_rpm = round(base_rpm - diff, 1)
        return left_rpm, right_rpm


def main(args=None):
    rclpy.init(args=args)
    node = MotorController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
