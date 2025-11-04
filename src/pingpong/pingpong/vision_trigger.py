import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import time

class VisionTriggerTest(Node):
    def __init__(self):
        super().__init__('vision_trigger_test')
        self.pub_age = self.create_publisher(Bool, '/age_trigger', 10)
        self.pub_body = self.create_publisher(Bool, '/body_trigger', 10)
        self.pub_serve = self.create_publisher(Bool, '/serve_trigger', 10)

    def test_loop(self):
        while rclpy.ok():
            cmd = input("Enter command (a=age, b=body, s=serve, q=quit): ")
            if cmd == 'a':
                self.pub_age.publish(Bool(data=True))
                time.sleep(0.1)
                self.pub_age.publish(Bool(data=False))
            elif cmd == 'b':
                self.pub_body.publish(Bool(data=True))
                time.sleep(0.1)
                self.pub_body.publish(Bool(data=False))
            elif cmd == 's':
                self.pub_serve.publish(Bool(data=True))
                time.sleep(0.1)
                self.pub_serve.publish(Bool(data=False))
            elif cmd == 'q':
                break

def main():
    rclpy.init()
    node = VisionTriggerTest()
    node.test_loop()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
