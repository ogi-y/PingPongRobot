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
        self.declare_parameter('court. width', 1525.0)
        self.declare_parameter('court.length', 2740.0)
        self.declare_parameter('strategy_mode', 'adaptive')  # 【追加】戦略モード
        
        # 値の読み込み
        self.court_w = self.get_parameter('court.width').value
        self. court_l = self.get_parameter('court.length').value
        self.strategy_mode = self.get_parameter('strategy_mode').value
        
        # ターゲット座標の定義
        self.coord_l = self.court_w * 0.2  # 左サイド
        self.coord_c = self.court_w * 0.5  # センター
        self.coord_r = self.court_w * 0.8  # 右サイド
        
        self.coord_short = 500.0   # ネット際
        self.coord_mid   = 2000.0  # ミドル
        self.coord_deep  = 2600.0  # エンドライン際

        self.get_logger().info(f"Strategy AI Ready.  Mode: {self.strategy_mode}")

        # --- データ保持用の変数 ---
        self. latest_age = 30
        self.latest_people = []
        self.age_updated = False
        self.pose_updated = False

        # --- 通信設定 ---
        self.create_subscription(String, '/age/estimation_result', self.age_callback, 10)
        self.create_subscription(String, '/pose/detected_positions', self.pose_callback, 10)
        self.create_subscription(Bool, '/serve_trigger', self.trigger_callback, 10)
        
        self.client = self.create_client(TargetShot, 'target_shot')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for Ballistics Node...')

    def age_callback(self, msg):
        try:
            data = json.loads(msg.data)
            if isinstance(data, list) and len(data) > 0:
                self.latest_age = data[0]. get('age', 30)
                self.age_updated = True
                self. get_logger().debug(f"Age updated:  {self.latest_age}")
        except Exception as e:
            self.get_logger().warn(f"Failed to parse age data: {e}")

    def pose_callback(self, msg):
        try:
            data = json.loads(msg.data)
            if isinstance(data, list):
                self.latest_people = data
                self.pose_updated = True
                self.get_logger().debug(f"Pose updated: {len(data)} people")
        except Exception as e:
            self. get_logger().warn(f"Failed to parse pose data: {e}")

    def trigger_callback(self, msg):
        if msg.data:
            self. think_and_order()

    # 【新規】画像座標をコート座標に変換
    def image_x_to_court_x(self, img_x):
        """
        画像のX座標（0-640）をコート座標（0-1525）に変換
        """
        if img_x is None:
            return self.coord_c
        # 線形変換
        return (img_x / CAMERA_WIDTH) * self.court_w

    # 【新規】モード別戦略決定
    def decide_strategy_adaptive(self, player_x_img, age):
        """現在の賢いモード（逆を狙う）"""
        target_x = self.coord_c
        target_y = self.coord_mid
        speed = 15
        spin = 0
        
        # 年齢によるレベル分け
        level = 1
        if age < 10:  level = 0
        elif age < 30: level = 1
        elif age < 40: level = 2
        else: level = 3

        img_center = CAMERA_WIDTH / 2
        
        if player_x_img is None: 
            target_x = random.choice([self.coord_l, self.coord_c, self.coord_r])
            target_y = random.choice([self.coord_mid, self.coord_deep])
        else:
            # 逆を狙う
            if player_x_img < img_center:
                target_x = self.coord_r
                spin_direction = 1
            else: 
                target_x = self.coord_l
                spin_direction = -1

            if level == 0:
                # 手加減モード
                if player_x_img < img_center:
                    target_x = self.coord_l
                else:
                    target_x = self.coord_r
                spin_direction = 0

            shot_type = random.random()
            if shot_type < 0.3 and level >= 2:
                target_y = self.coord_short
                speed = 30
                spin = 20
            else:
                target_y = self.coord_deep
                if level >= 3:
                    speed = 50
                    spin = 30 * spin_direction
                elif level == 2:
                    speed = 30
                    spin = 15 * spin_direction
                else:
                    speed = 10
                    spin = 0

        return target_x, target_y, speed, spin, level

    def decide_strategy_chase(self, player_x_img, age):
        """追跡モード - 常に相手を狙う"""
        if player_x_img is None: 
            target_x = self.coord_c
        else:
            # 相手の位置をそのまま狙う
            target_x = self.image_x_to_court_x(player_x_img)
        
        target_y = self.coord_deep
        speed = 15  # ゆっくり
        spin = 0    # 真っ直ぐ
        level = 0
        
        return target_x, target_y, speed, spin, level

    def decide_strategy_avoid(self, player_x_img, age):
        """回避モード - 常に最も遠い場所を狙う"""
        if player_x_img is None: 
            target_x = random.choice([self.coord_l, self.coord_r])
        else:
            court_x = self.image_x_to_court_x(player_x_img)
            # 左右どちらが遠いか判定
            dist_to_left = abs(court_x - 0)
            dist_to_right = abs(court_x - self.court_w)
            
            if dist_to_left > dist_to_right:
                target_x = self.coord_l * 0.5  # さらに左端
            else:
                target_x = self.coord_r + (self.court_w - self.coord_r) * 0.5  # さらに右端
        
        target_y = self.coord_deep
        
        # 年齢に応じたスピード
        if age < 20:
            speed = 20
            spin = 15
        elif age < 40:
            speed = 35
            spin = 25
        else:
            speed = 45
            spin = 30
        
        level = 2
        return target_x, target_y, speed, spin, level

    def decide_strategy_random(self, player_x_img, age):
        """完全ランダム"""
        target_x = random.uniform(self.coord_l, self.coord_r)
        target_y = random.uniform(self.coord_mid, self.coord_deep)
        speed = random.randint(10, 40)
        spin = random.randint(-20, 20)
        level = random.randint(0, 3)
        
        return target_x, target_y, speed, spin, level

    def decide_strategy_center(self, player_x_img, age):
        """センター固定 - 常に中央"""
        target_x = self.coord_c
        target_y = self. coord_mid
        speed = 12
        spin = 0
        level = 0
        
        return target_x, target_y, speed, spin, level

    def decide_strategy_beginner(self, player_x_img, age):
        """初心者モード - 超優しい"""
        if player_x_img is None: 
            target_x = self. coord_c
        else: 
            # 相手の位置に近い場所を狙う（やや同じ側）
            court_x = self.image_x_to_court_x(player_x_img)
            target_x = court_x * 0.8 + self.coord_c * 0.2  # 80%相手位置、20%中央
        
        target_y = self.coord_mid  # 手前
        speed = 8   # 非常にゆっくり
        spin = 0    # スピンなし
        level = 0
        
        return target_x, target_y, speed, spin, level

    def think_and_order(self):
        # --- 1. データ取得 ---
        player_x_img = None
        age = self.latest_age
        
        if self.latest_people and len(self.latest_people) > 0:
            player_x_img = self.latest_people[0].get("pos_x")
            self.get_logger().info(f"Player detected at X={player_x_img}, Age={age}")
        else:
            self.get_logger().info(f"No player detected, Age={age}")

        # --- 2. モード別戦略決定 ---
        # パラメータを動的に読み込み（GUIから変更可能にするため）
        self.strategy_mode = self.get_parameter('strategy_mode').value
        
        if self.strategy_mode == 'chase':
            target_x, target_y, speed, spin, level = self.decide_strategy_chase(player_x_img, age)
        elif self.strategy_mode == 'avoid':
            target_x, target_y, speed, spin, level = self.decide_strategy_avoid(player_x_img, age)
        elif self.strategy_mode == 'random': 
            target_x, target_y, speed, spin, level = self.decide_strategy_random(player_x_img, age)
        elif self.strategy_mode == 'center':
            target_x, target_y, speed, spin, level = self.decide_strategy_center(player_x_img, age)
        elif self.strategy_mode == 'beginner':
            target_x, target_y, speed, spin, level = self.decide_strategy_beginner(player_x_img, age)
        else:  # 'adaptive' or default
            target_x, target_y, speed, spin, level = self.decide_strategy_adaptive(player_x_img, age)

        roll = 0.0

        # --- 3. 軌道計算機への指令送信 ---
        self.get_logger().info(
            f"AI Decision [{self.strategy_mode. upper()}]: "
            f"Target({target_x:.0f}, {target_y:.0f}) "
            f"Speed:{speed} Spin:{spin} Level:{level}"
        )

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