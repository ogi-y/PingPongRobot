import rclpy
from rclpy. node import Node
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
        
        # 値の読み込み
        self.court_w = self.get_parameter('court.width').value
        self.court_l = self. get_parameter('court.length').value
        
        # ターゲット座標の定義
        self.coord_l = self.court_w * 0.2  # 左サイド
        self.coord_c = self.court_w * 0.5  # センター
        self.coord_r = self.court_w * 0.8  # 右サイド
        
        self.coord_short = 500.0   # ネット際
        self.coord_mid   = 2000.0  # ミドル
        self.coord_deep  = 2600.0  # エンドライン際

        self.get_logger().info("Strategy AI Ready.  Waiting for triggers...")

        # --- 【修正】データ保持用の変数 ---
        self. latest_age = 30          # キャッシュされた年齢
        self. latest_people = []       # キャッシュされた人物位置
        self.age_updated = False      # データ更新フラグ
        self.pose_updated = False

        # --- 【修正】通信設定：2つのトピックをsubscribe ---
        self.create_subscription(String, '/age/estimation_result', self.age_callback, 10)
        self.create_subscription(String, '/pose/detected_positions', self.pose_callback, 10)
        self.create_subscription(Bool, '/serve_trigger', self.trigger_callback, 10)
        
        self.client = self.create_client(TargetShot, 'target_shot')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for Ballistics Node...')

    # --- 【新規】年齢データのコールバック ---
    def age_callback(self, msg):
        """
        age_estimatorからのデータを受信
        データ形式:  [{"id": 0, "age": 25, "bbox":  {...}}]
        """
        try:
            data = json. loads(msg.data)
            if isinstance(data, list) and len(data) > 0:
                # 最初の顔の年齢を使用
                self.latest_age = data[0]. get('age', 30)
                self.age_updated = True
                self. get_logger().debug(f"Age updated: {self.latest_age}")
        except Exception as e:
            self. get_logger().warn(f"Failed to parse age data: {e}")

    # --- 【新規】姿勢データのコールバック ---
    def pose_callback(self, msg):
        """
        pose_estimatorからのデータを受信
        データ形式: [{"id":  0, "pos_x": 320, "pos_y": 240, "part_used": "Hand", ... }]
        """
        try:
            data = json.loads(msg.data)
            if isinstance(data, list):
                self.latest_people = data
                self.pose_updated = True
                self.get_logger().debug(f"Pose updated: {len(data)} people")
        except Exception as e:
            self.get_logger().warn(f"Failed to parse pose data: {e}")

    def trigger_callback(self, msg):
        if msg.data:
            self.think_and_order()

    def think_and_order(self):
        # --- 1. 状況判断（統合されたデータから取得） ---
        player_x_img = None
        age = self.latest_age  # キャッシュから取得
        
        # 人物位置の取得
        if self.latest_people and len(self.latest_people) > 0:
            # 最も信頼度の高い人（リスト先頭）を使用
            player_x_img = self.latest_people[0]. get("pos_x")
            self.get_logger().info(f"Player detected at X={player_x_img}, Age={age}")
        else:
            self.get_logger().info(f"No player detected, Age={age}")

        # --- 2. 戦略決定ロジック（変更なし） ---
        target_x = self.coord_c
        target_y = self.coord_mid
        speed = 15
        spin = 0
        roll = 0.0
        
        # 年齢によるレベル分け
        level = 1
        if age < 10:  
            level = 0
        elif age < 30: 
            level = 1
        elif age < 40: 
            level = 2
        else: 
            level = 3

        # プレイヤー位置によるコース打ち分け
        img_center = CAMERA_WIDTH / 2
        
        if player_x_img is None:
            # いない場合はランダム
            target_x = random. choice([self.coord_l, self.coord_c, self.coord_r])
            target_y = random.choice([self.coord_mid, self.coord_deep])
        else:
            # プレイヤーの位置に応じて逆を狙う
            if player_x_img < img_center:  
                target_x = self.coord_r
                spin_direction = 1
            else: 
                target_x = self. coord_l
                spin_direction = -1

            # レベル補正
            if level == 0:
                # 手加減モード
                if player_x_img < img_center:  
                    target_x = self.coord_l
                else:  
                    target_x = self.coord_r
                spin_direction = 0

            # 深さと球速の決定
            shot_type = random.random()
            
            if shot_type < 0.3 and level >= 2:
                # ドロップショット
                target_y = self.coord_short
                speed = 30
                spin = 20
            else:
                # ドライブ/スマッシュ
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

        # --- 3. 軌道計算機への指令送信 ---
        self.get_logger().info(
            f"AI Decision: Target({target_x:. 0f}, {target_y:.0f}) "
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