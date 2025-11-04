import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, String, Bool
import random
import json

class ServeController(Node):
    def __init__(self):
        super().__init__('serve_controller')
        self.age = None
        self.player_pos = None
        self.level = 0
        self.triggered = False
        self.robo_pos = "center"

        self.create_subscription(String, '/age', self.age_callback, 10)
        # /age topic JSON example:
        # {"ages": [{"id": 1, "age": 25}, {"id": 2, "age": 30}]}
        self.create_subscription(String, '/player_pos', self.player_pos_callback, 10)
        # /player_pos topic JSON example:
        # {"pos": "left"}
        self.create_subscription(Bool, '/serve_trigger', self.serve_callback, 10)
        self.publisher = self.create_publisher(String, '/serve_param', 10)
        self.timer = self.create_timer(0.1, self.decide_serve)

        self.robo_pos_choices = {
                "left": ["left", "center"],
                "center": ["left", "right", "center"],
                "right": ["right", "center"],
            }

    def age_callback(self, msg):
        try:
            data = json.loads(msg.data)
            if not data["ages"]:
                return
            # 今は最初の人の年齢を使用
            self.age = data["ages"][0]["age"]
        except Exception as e:
            self.get_logger().error(f"Failed to parse age data: {e}")

    def player_pos_callback(self, msg):
        try:
            data = json.loads(msg.data)
            if "pos" not in data:
                return
            self.player_pos = data["pos"]
        except Exception as e:
            self.get_logger().error(f"Failed to parse player position data: {e}")

    def serve_callback(self, msg):
        self.triggered = msg.data
        if self.triggered:
            self.get_logger().info("Fire!")

    def decide_serve(self):
        if not self.triggered:
            return
        if self.age is not None:
            age = self.age
            if age < 20:
                self.level = 0
            elif age < 40:
                self.level = 1
            else:
                self.level = 2
        else:
            self.level = random.choice([0, 1, 2])
            self.get_logger().info("Age not available, random level selected")
        
        if self.player_pos is not None:
            player_pos = self.player_pos
        else:
            player_pos = random.choice(["left", "right", "center"])
            self.get_logger().info("Player position not available, random position selected")

        if self.level == 0:
            speed, spin = 0.3, 0.2
            course_x = player_pos
            course_y = "middle"
            spin_dir = "top"
            serve_pos = "center"
        elif self.level == 1:
            speed, spin = 0.6, 0.5
            course_x = random.choice(["left", "right", "center"])
            course_y = random.choice(["middle", "back"])
            spin_dir = random.choice(["top", "back", "left", "right"])
            serve_pos = random.choice(self.robo_pos_choices[self.robo_pos])
        else:
            speed, spin = 0.9, 0.8
            course_x_choices = [pos for pos in ["left", "right", "center"] if pos != player_pos]
            course_x = random.choice(course_x_choices)
            course_y = random.choice(["front", "middle", "back"])
            spin_dir = random.choice(["top", "back", "left", "right"])
            serve_pos = random.choice(["left", "right", "center"])

        speed += random.uniform(-0.05, 0.05)
        spin += random.uniform(-0.05, 0.05)
        self.robo_pos = serve_pos

        serve_param = {
            "speed": round(speed, 2),
            "spin": round(spin, 2),
            "spin_dir": spin_dir,
            "course_x": course_x,
            "course_y": course_y,
            "serve_pos": serve_pos,
        }

        msg = String()
        msg.data = json.dumps(serve_param)
        self.publisher.publish(msg)

        self.get_logger().info(f"Age:{self.age}, Player Position:{self.player_pos}, Level:{self.level}, Serve parameters: {serve_param}")
        self.triggered = False

def main(args=None):
    rclpy.init(args=args)
    node = ServeController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
