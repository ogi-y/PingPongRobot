import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from pingpong_msgs.srv import TargetShot # 新しいサービス
import json
import random

CAMERA_WIDTH = 640
COURT_WIDTH = 1525.0
COURT_LENGTH = 2740.0

# コート座標定義 (ロボットから見た相対座標 mm)
# X: 0(左端) ~ 2000(右端)
# Y: 0(手前) ~ 2740(奥端)
COORD_L = COURT_WIDTH * 0.25
COORD_C = COURT_WIDTH * 0.5
COORD_R = COURT_WIDTH * 0.75

COORD_SHT = COURT_LENGTH * 0.6
COORD_MID = COURT_LENGTH * 0.75
COORD_LNG = COURT_LENGTH * 0.9

class StrategyServer(Node):
    def __init__(self):
        super().__init__('strategy_server')
        self.create_subscription(String, '/vision/analysis', self.vision_callback, 10)
        self.create_subscription(Bool, '/serve_trigger', self.trigger_callback, 10)

        # 軌道計算機へのクライアント
        self.client = self.create_client(TargetShot, 'target_shot')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for Ballistics Node...')

        self.latest_vision_data = None
        self.get_logger().info('Strategy AI Ready.')

    def vision_callback(self, msg):
        try:
            self.latest_vision_data = json.loads(msg.data)
        except:
            pass

    def trigger_callback(self, msg):
        if msg.data:
            self.think_and_order()

    def think_and_order(self):
        # 1. 状況判断 (どこに人がいるか？)
        player_x_img = None
        age = 30
        
        if self.latest_vision_data:
            age = int(self.latest_vision_data.get("age", 30))
            people = self.latest_vision_data.get("people", [])
            if people:
                player_x_img = people[0].get("pos_x")

        # 2. ターゲット座標の決定 (戦略レイヤー)
        target_x = COORD_C
        target_y = COORD_MID
        speed_mode = "normal"
        level = 1

        # 年齢による手加減
        if age < 10:
            level = 0
        elif age < 20:
            level = 1
        elif age < 40:
            level = 2
        else:
            level = 3

        # 位置による左右の打ち分け
        if player_x_img is None:
            # いないならランダム
            target_x = random.choice([COORD_L, COORD_C, COORD_R])
            speed_mode = random.choice(["slow", "normal"])
        else:
            # オープンスペース攻撃
            if player_x_img < (CAMERA_WIDTH / 3): # 敵は左
                if level <= 1:
                    target_x = COORD_L
                else:
                    target_x = random.choice([COORD_C, COORD_R])
            elif player_x_img > (CAMERA_WIDTH * 2 / 3): # 敵は右
                if level <= 1:
                    target_x = COORD_R
                else:
                    target_x = random.choice([COORD_L, COORD_C])
            else:
                if level <= 1:
                    target_x = COORD_C
                else:
                    target_x = random.choice([COORD_L, COORD_R])
            # レベルによる速さ調整
            if level == 0:
                speed_mode = "slow"
                target_y = COORD_MID
            elif level == 1:
                speed_mode = random.choice(["slow", "normal"])
                target_y = random.choice([COORD_MID, COORD_LNG])
            else:
                speed_mode = random.choice(["slow","fast"])
                target_y = random.choice([COORD_SHT, COORD_MID, COORD_LNG])
            
        # 3. 軌道計算機への指令
        self.get_logger().info(f"Order: Hit ({target_x:.0f}, {target_y:.0f}) mode={speed_mode}")
        
        req = TargetShot.Request()
        req.target_x = float(target_x)
        req.target_y = float(target_y)
        req.height_z = 0.0 # テーブル面狙い
        req.speed_mode = speed_mode

        future = self.client.call_async(req)

def main(args=None):
    rclpy.init(args=args)
    node = StrategyServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()