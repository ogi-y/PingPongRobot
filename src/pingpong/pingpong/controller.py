import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, String
import random
import json

class ServeController(Node):
    def __init__(self):
        super().__init__('serve_controller')
        self.age = None
        self.body_position = None
        self.level = 0

        self.create_subscription(Int32, '/age', self.age_callback, 10)
        self.create_subscription(String, '/body_position', self.body_callback, 10)
        self.publisher = self.create_publisher(String, '/serve_param', 10)

        self.timer = self.create_timer(0.5, self.decide_serve)

    def age_callback(self, msg):
        self.age = msg.data

    def body_callback(self, msg):
        self.body_position = msg.data

    def decide_serve(self):
        if self.age_level is None or self.body_position is None:
            return

        # 年齢レベルによる基本値
        if self.age_level == 0:
            speed, spin = 0.3, 0.2
        elif self.age_level == 1:
            speed, spin = 0.6, 0.5
        else:
            speed, spin = 0.9, 0.8

        # 体位置によるコースと回転方向
        if self.body_position == "left":
            course_x, spin_dir = "right", "side_right"
        elif self.body_position == "right":
            course_x, spin_dir = "left", "side_left"
        else:
            course_x, spin_dir = "center", "top"

        # 奥行きコースをランダムに
        course_y = random.choice(["front", "middle", "back"])

        # 少しランダム補正
        speed += random.uniform(-0.05, 0.05)
        spin += random.uniform(-0.05, 0.05)

        serve_param = {
            "speed": round(speed, 2),
            "spin": round(spin, 2),
            "spin_dir": spin_dir,
            "course_x": course_x,
            "course_y": course_y,
        }

        msg = String()
        msg.data = json.dumps(serve_param)
        self.publisher.publish(msg)

        self.get_logger().info(f"Serve parameters: {serve_param}")

def main(args=None):
    rclpy.init(args=args)
    node = ServeController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
