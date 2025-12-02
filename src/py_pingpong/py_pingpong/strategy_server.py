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
        
        # --- パラメータの宣言と取得 ---
        # (デフォルト値を設定しておくとYAML読み込み失敗時も安心です)
        self.declare_parameter('court.width', 1525.0)
        self.declare_parameter('court.length', 2740.0)
        
        self.declare_parameter('coordinates.x_left_ratio', 0.2)
        self.declare_parameter('coordinates.x_center_ratio', 0.5)
        self.declare_parameter('coordinates.x_right_ratio', 0.8)
        
        self.declare_parameter('coordinates.y_short_ratio', 0.6)
        self.declare_parameter('coordinates.y_middle_ratio', 0.75)
        self.declare_parameter('coordinates.y_long_ratio', 0.9)

        # 値の読み込み
        court_w = self.get_parameter('court.width').value
        court_l = self.get_parameter('court.length').value
        
        # 座標計算 (クラスメンバ変数として保持)
        self.coord_l = court_w * self.get_parameter('coordinates.x_left_ratio').value
        self.coord_c = court_w * self.get_parameter('coordinates.x_center_ratio').value
        self.coord_r = court_w * self.get_parameter('coordinates.x_right_ratio').value
        
        self.coord_sht = court_l * self.get_parameter('coordinates.y_short_ratio').value
        self.coord_mid = court_l * self.get_parameter('coordinates.y_middle_ratio').value
        self.coord_lng = court_l * self.get_parameter('coordinates.y_long_ratio').value

        self.get_logger().info(f"Config Loaded: Width={court_w}, Length={court_l}")

        # --- 通信設定 ---
        self.create_subscription(String, '/vision/analysis', self.vision_callback, 10)
        self.create_subscription(Bool, '/serve_trigger', self.trigger_callback, 10)
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
        target_x = self.coord_c
        target_y = self.coord_mid
        speed_mode = "normal"
        spin = 0
        roll = 0.0
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
            target_x = random.choice([self.coord_l, self.coord_c, self.coord_r])
            target_y = random.choice([self.coord_sht, self.coord_mid, self.coord_lng])
            speed_mode = random.choice(["slow", "normal"])
        else:
            # オープンスペース攻撃
            if player_x_img < (CAMERA_WIDTH / 3): # 敵は左
                if level <= 1:
                    target_x = self.coord_l
                else:
                    target_x = random.choice([self.coord_c, self.coord_r])
            elif player_x_img > (CAMERA_WIDTH * 2 / 3): # 敵は右
                if level <= 1:
                    target_x = self.coord_r
                else:
                    target_x = random.choice([self.coord_l, self.coord_c])
            else:
                if level <= 1:
                    target_x = self.coord_c
                else:
                    target_x = random.choice([self.coord_l, self.coord_r])
            # レベルによる速さ調整
            if level == 0:
                speed_mode = "slow"
                target_y = self.coord_mid
            elif level == 1:
                speed_mode = random.choice(["slow", "normal"])
                target_y = random.choice([self.coord_mid, self.coord_lng])
                spin = random.choice([0, 5, -5])
                roll = random.uniform(-90, 90)
            else:
                speed_mode = random.choice(["slow","fast"])
                target_y = random.choice([self.coord_sht, self.coord_mid, self.coord_lng])
                spin = random.choice([0, 5, 10, -5, -10])
                roll = random.uniform(-90, 90)
            
        # 3. 軌道計算機への指令
        self.get_logger().info(f"Order: Hit ({target_x:.0f}, {target_y:.0f}) mode={speed_mode}")

        req = TargetShot.Request()
        req.target_x = float(target_x)
        req.target_y = float(target_y)
        req.height_z = 0.0 # テーブル面狙い
        req.speed_mode = speed_mode
        req.spin = spin
        req.roll = roll

        future = self.client.call_async(req)

def main(args=None):
    rclpy.init(args=args)
    node = StrategyServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()