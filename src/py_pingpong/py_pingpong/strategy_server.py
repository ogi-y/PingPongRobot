import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from pingpong_msgs.srv import TargetShot
import json
import random

# 画像サイズとコートサイズ定義
CAMERA_WIDTH = 640
COURT_WIDTH = 1525.0
COURT_LENGTH = 2740.0

class StrategyServer(Node):
    def __init__(self):
        super().__init__('strategy_server')
        
        # --- パラメータ設定 ---
        # コートの狙うべき座標の目安 (左・中・右 / 手前・中・奥)
        self.declare_parameter('court.width', 1525.0)
        self.declare_parameter('court.length', 2740.0)
        
        # 値の読み込み
        self.court_w = self.get_parameter('court.width').value
        self.court_l = self.get_parameter('court.length').value
        
        # ターゲット座標の定義
        # X座標: 0=左端, 762.5=中央, 1525=右端
        self.coord_l = self.court_w * 0.2  # 左サイド
        self.coord_c = self.court_w * 0.5  # センター
        self.coord_r = self.court_w * 0.8  # 右サイド
        
        # Y座標: 0=手前, 2740=奥
        self.coord_short = 500.0   # ネット際 (バウンド狙い)
        self.coord_mid   = 2000.0  # ミドル
        self.coord_deep  = 2600.0  # エンドライン際

        self.get_logger().info("Strategy AI Ready. Waiting for triggers...")

        # --- 通信設定 ---
        self.create_subscription(String, '/vision/analysis', self.vision_callback, 10)
        self.create_subscription(Bool, '/serve_trigger', self.trigger_callback, 10)
        
        self.client = self.create_client(TargetShot, 'target_shot')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for Ballistics Node...')
            
        self.latest_vision_data = None

    def vision_callback(self, msg):
        try:
            self.latest_vision_data = json.loads(msg.data)
        except:
            pass

    def trigger_callback(self, msg):
        if msg.data:
            self.think_and_order()

    def think_and_order(self):
        # --- 1. 状況判断 (Visionデータの解析) ---
        player_x_img = None
        age = 30
        
        if self.latest_vision_data:
            age = int(self.latest_vision_data.get("age", 30))
            people = self.latest_vision_data.get("people", [])
            # 最も信頼度の高い人を探す（単純にリスト先頭でもOK）
            if people:
                player_x_img = people[0].get("pos_x")

        # --- 2. 戦略決定ロジック ---
        
        # デフォルト値
        target_x = self.coord_c
        target_y = self.coord_mid
        speed = 15       # 標準パワー
        spin = 0
        roll = 0.0
        
        # 年齢によるレベル分け (0:手加減 ~ 3:本気)
        level = 1
        if age < 10: level = 0
        elif age < 30: level = 1
        elif age < 40: level = 2
        else: level = 3

        # プレイヤー位置によるコース打ち分け (逆を突く)
        img_center = CAMERA_WIDTH / 2
        
        if player_x_img is None:
            # いない場合はランダム
            target_x = random.choice([self.coord_l, self.coord_c, self.coord_r])
            target_y = random.choice([self.coord_mid, self.coord_deep])
        
        else:
            # プレイヤーの位置に応じて逆を狙う
            if player_x_img < img_center: 
                target_x = self.coord_r # 空いている右側を狙う
                spin_direction = 1      # 右回転で逃がす
            else:
                # 敵は右にいる -> 左を狙う
                target_x = self.coord_l
                spin_direction = -1     # 左回転で逃がす

            # レベル補正
            if level == 0:
                # 手加減: 逆にプレイヤーの正面に打ってあげる
                if player_x_img < img_center: target_x = self.coord_l
                else: target_x = self.coord_r
                spin_direction = 0

            # 深さと球速の決定 (ランダムミックス)
            # 30%の確率でドロップショット、70%でドライブ
            shot_type = random.random()
            
            if shot_type < 0.3 and level >= 2:
                # ドロップショット (手前・遅く・下回転)
                target_y = self.coord_short
                speed = 30
                spin = 20
            else:
                # ドライブ/スマッシュ (奥・速く・横回転)
                target_y = self.coord_deep
                
                # レベルが高いほど速く、カーブがきつい
                if level >= 3:
                    speed = 50
                    spin = 30 * spin_direction
                elif level == 2:
                    speed = 30
                    spin = 15 * spin_direction
                else: # level 0, 1
                    speed = 10
                    spin = 0

        # --- 3. 軌道計算機への指令送信 ---
        self.get_logger().info(f"AI Decision: Hit({target_x:.0f}, {target_y:.0f}) Spd:{speed} Spin:{spin}")

        req = TargetShot.Request()
        req.target_x = float(target_x)
        req.target_y = float(target_y)
        req.height_z = 0.0
        req.roll_deg = float(roll)
        req.spin = int(spin)
        req.speed = int(speed)

        future = self.client.call_async(req)

def main(args=None):
    rclpy.init(args=args)
    node = StrategyServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()