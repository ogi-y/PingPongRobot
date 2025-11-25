import sys
import rclpy
from rclpy.node import Node
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QSlider, QComboBox, QGroupBox, 
                             QTabWidget, QSpinBox, QCheckBox)
from PyQt5.QtCore import Qt, QTimer
import random

# ROSメッセージ/サービスのインポート
from pingpong_msgs.srv import TargetShot # セミオート用 (座標指定)
from pingpong_msgs.srv import Shoot      # ★追加: テンプレート用 (難易度指定)
from pingpong_msgs.msg import ShotParams # フルマニュアル用
from std_msgs.msg import Bool

class PingPongGUI(QWidget):
    def __init__(self, ros_node):
        super().__init__()
        self.node = ros_node
        
        # Strategy AI用タイマー
        self.auto_timer = QTimer()
        self.auto_timer.timeout.connect(self.send_auto_trigger)
        
        # Template連射用タイマー
        self.template_timer = QTimer()
        self.template_timer.timeout.connect(self.fire_template_service)
        
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Ping Pong Robot Commander')
        self.setGeometry(100, 100, 650, 750)
        
        main_layout = QVBoxLayout()
        
        # --- タブウィジェットの作成 ---
        tabs = QTabWidget()
        tabs.addTab(self.create_auto_tab(), "AUTO (Strategy)")      # カメラ戦略
        tabs.addTab(self.create_template_tab(), "AUTO (Template)")  # ★これ: 難易度指定
        tabs.addTab(self.create_semiauto_tab(), "SEMI-AUTO (Target)") # 座標指定
        tabs.addTab(self.create_manual_tab(), "FULL MANUAL (Motors)") # 直接制御
        
        main_layout.addWidget(tabs)

        # --- 共通ステータスバー ---
        self.label_status = QLabel("Status: Ready")
        self.label_status.setAlignment(Qt.AlignCenter)
        self.label_status.setStyleSheet("font-size: 14px; font-weight: bold; color: #333; padding: 10px;")
        main_layout.addWidget(self.label_status)

        self.setLayout(main_layout)

    # ----------------------------------------------------------------
    # 1. AUTO (Strategy) モード
    # ----------------------------------------------------------------
    def create_auto_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        lbl = QLabel("AI Strategy Mode: Uses Camera & Vision.\nRobot decides based on player position.")
        lbl.setStyleSheet("font-weight: bold; color: #4CAF50;")
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        # 間隔設定
        group_interval = QGroupBox("Firing Interval")
        layout_int = QHBoxLayout()
        layout_int.addWidget(QLabel("Interval (sec):"))
        self.spin_interval = QSpinBox()
        self.spin_interval.setRange(2, 10)
        self.spin_interval.setValue(3)
        layout_int.addWidget(self.spin_interval)
        group_interval.setLayout(layout_int)
        layout.addWidget(group_interval)

        # スタート/ストップ
        self.btn_auto_toggle = QPushButton("START STRATEGY AI")
        self.btn_auto_toggle.setFixedHeight(80)
        self.btn_auto_toggle.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; font-size: 18px;")
        self.btn_auto_toggle.setCheckable(True)
        self.btn_auto_toggle.clicked.connect(self.toggle_auto_fire)
        layout.addWidget(self.btn_auto_toggle)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def toggle_auto_fire(self):
        if self.btn_auto_toggle.isChecked():
            interval_ms = self.spin_interval.value() * 1000
            self.auto_timer.start(interval_ms)
            self.btn_auto_toggle.setText("STOP STRATEGY AI")
            self.btn_auto_toggle.setStyleSheet("background-color: #F44336; color: white; font-weight: bold; font-size: 18px;")
            self.label_status.setText(f"Status: Strategy AI running ({self.spin_interval.value()}s)...")
        else:
            self.auto_timer.stop()
            self.btn_auto_toggle.setText("START STRATEGY AI")
            self.btn_auto_toggle.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; font-size: 18px;")
            self.label_status.setText("Status: Strategy AI Stopped.")

    def send_auto_trigger(self):
        self.node.publish_trigger()

    # ----------------------------------------------------------------
    # ★ 2. AUTO (Template) モード - Controller内蔵テンプレート呼び出し
    # ----------------------------------------------------------------
    def create_template_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        lbl = QLabel("Template Mode: Controller's internal presets.\nSelect multiple levels to mix difficulties.")
        lbl.setStyleSheet("font-weight: bold; color: #9C27B0;")
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        # 難易度選択 (チェックボックスに変更)
        group_level = QGroupBox("Difficulty Levels (Mixable)")
        layout_level = QHBoxLayout()
        
        # IDはController側の定義に合わせる (1:Easy, 2:Normal, 3:Hard, 4:Pro)
        self.chk_lv1 = QCheckBox("Lv1: Easy")
        self.chk_lv2 = QCheckBox("Lv2: Normal")
        self.chk_lv3 = QCheckBox("Lv3: Hard")
        self.chk_lv4 = QCheckBox("Lv4: Pro")
        
        # デフォルトでNormalをONにしておく
        self.chk_lv2.setChecked(True)

        layout_level.addWidget(self.chk_lv1)
        layout_level.addWidget(self.chk_lv2)
        layout_level.addWidget(self.chk_lv3)
        layout_level.addWidget(self.chk_lv4)
        group_level.setLayout(layout_level)
        layout.addWidget(group_level)

        # 間隔設定
        group_interval = QGroupBox("Firing Interval")
        layout_int = QHBoxLayout()
        layout_int.addWidget(QLabel("Interval (sec):"))
        self.spin_tmpl_interval = QSpinBox()
        self.spin_tmpl_interval.setRange(2, 10)
        self.spin_tmpl_interval.setValue(3)
        layout_int.addWidget(self.spin_tmpl_interval)
        group_interval.setLayout(layout_int)
        layout.addWidget(group_interval)

        # スタート/ストップ
        self.btn_tmpl_toggle = QPushButton("START TEMPLATE FIRE")
        self.btn_tmpl_toggle.setFixedHeight(80)
        self.btn_tmpl_toggle.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold; font-size: 18px;")
        self.btn_tmpl_toggle.setCheckable(True)
        self.btn_tmpl_toggle.clicked.connect(self.toggle_template_fire)
        layout.addWidget(self.btn_tmpl_toggle)

        # 単発テスト
        btn_single = QPushButton("Fire Single Shot (Test)")
        btn_single.clicked.connect(self.fire_template_service)
        layout.addWidget(btn_single)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def toggle_template_fire(self):
        if self.btn_tmpl_toggle.isChecked():
            # チェックが一つもない場合のガード
            if not any([self.chk_lv1.isChecked(), self.chk_lv2.isChecked(), 
                        self.chk_lv3.isChecked(), self.chk_lv4.isChecked()]):
                self.label_status.setText("Status: Error! Select at least one level.")
                self.btn_tmpl_toggle.setChecked(False)
                return

            interval_ms = self.spin_tmpl_interval.value() * 1000
            self.template_timer.start(interval_ms)
            self.btn_tmpl_toggle.setText("STOP TEMPLATE FIRE")
            self.btn_tmpl_toggle.setStyleSheet("background-color: #F44336; color: white; font-weight: bold; font-size: 18px;")
            self.label_status.setText(f"Status: Template Firing (Mixed Levels)...")
        else:
            self.template_timer.stop()
            self.btn_tmpl_toggle.setText("START TEMPLATE FIRE")
            self.btn_tmpl_toggle.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold; font-size: 18px;")
            self.label_status.setText("Status: Template Stopped.")

    def fire_template_service(self):
        """Shootサービスを呼び出す"""
        # 1. チェックされている難易度をリストアップ
        active_levels = []
        if self.chk_lv1.isChecked(): active_levels.append(1)
        if self.chk_lv2.isChecked(): active_levels.append(2)
        if self.chk_lv3.isChecked(): active_levels.append(3)
        if self.chk_lv4.isChecked(): active_levels.append(4)
        
        if not active_levels:
            self.label_status.setText("Status: No level selected!")
            return

        # 2. その中からランダムに1つ選ぶ
        selected_difficulty = random.choice(active_levels)
        
        self.label_status.setText(f"Status: Calling /shoot (Diff:{selected_difficulty})")
        self.node.send_shoot_template(selected_difficulty)

    # ----------------------------------------------------------------
    # 3. SEMI-AUTO モード (座標指定)
    # ----------------------------------------------------------------
    def create_semiauto_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        lbl = QLabel("Target Mode: Specify coordinates manually.")
        layout.addWidget(lbl)

        # スライダー群 (省略せず記述)
        self.semi_x, _ = self.create_slider("Target X (Width)", 0, 1525, 762)
        layout.addLayout(self.semi_x)
        self.semi_y, _ = self.create_slider("Target Y (Depth)", 0, 2740, 2200)
        layout.addLayout(self.semi_y)
        self.semi_roll, _ = self.create_slider("Head Roll (deg)", -45, 45, 0)
        layout.addLayout(self.semi_roll)
        self.semi_spin, _ = self.create_slider("Spin Power (-50~50)", -50, 50, 0)
        layout.addLayout(self.semi_spin)

        layout_spd = QHBoxLayout()
        layout_spd.addWidget(QLabel("Speed Mode:"))
        self.semi_speed = QComboBox()
        self.semi_speed.addItems(["slow", "normal", "fast"])
        self.semi_speed.setCurrentIndex(1)
        layout_spd.addWidget(self.semi_speed)
        layout.addLayout(layout_spd)

        btn_fire = QPushButton("FIRE TARGET SHOT")
        btn_fire.setFixedHeight(60)
        btn_fire.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        btn_fire.clicked.connect(self.fire_semiauto)
        layout.addWidget(btn_fire)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def fire_semiauto(self):
        x = float(self.semi_x.itemAt(1).widget().value())
        y = float(self.semi_y.itemAt(1).widget().value())
        roll = float(self.semi_roll.itemAt(1).widget().value())
        spin = int(self.semi_spin.itemAt(1).widget().value())
        speed = self.semi_speed.currentText()
        self.label_status.setText(f"Status: Semi-Auto -> ({x:.0f}, {y:.0f})")
        self.node.send_target_shot(x, y, roll, spin, speed)

    # ----------------------------------------------------------------
    # 4. FULL MANUAL モード
    # ----------------------------------------------------------------
    def create_manual_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        lbl = QLabel("Direct Control: Set raw values for all motors.")
        lbl.setStyleSheet("color: red;")
        layout.addWidget(lbl)

        self.man_pos, _ = self.create_slider("Robot Pos X (mm)", 0, 2000, 1000)
        layout.addLayout(self.man_pos)
        self.man_pitch, _ = self.create_slider("Pitch (deg)", -45, 45, 0)
        layout.addLayout(self.man_pitch)
        self.man_yaw, _ = self.create_slider("Yaw (deg)", -45, 45, 0)
        layout.addLayout(self.man_yaw)
        self.man_roll, _ = self.create_slider("Roll (deg)", -45, 45, 0)
        layout.addLayout(self.man_roll)
        self.man_pow_l, _ = self.create_slider("Power Left (-100-100)", -100, 100, 0)
        layout.addLayout(self.man_pow_l)
        self.man_pow_r, _ = self.create_slider("Power Right (-100-100)", -100, 100, 0)
        layout.addLayout(self.man_pow_r)

        btn_fire = QPushButton("FIRE RAW COMMAND")
        btn_fire.setFixedHeight(60)
        btn_fire.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        btn_fire.clicked.connect(self.fire_manual)
        layout.addWidget(btn_fire)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def fire_manual(self):
        msg = ShotParams()
        msg.pos = float(self.man_pos.itemAt(1).widget().value())
        msg.pitch_deg = float(self.man_pitch.itemAt(1).widget().value())
        msg.yaw_deg = float(self.man_yaw.itemAt(1).widget().value())
        msg.roll_deg = float(self.man_roll.itemAt(1).widget().value())
        msg.pow_left = int(self.man_pow_l.itemAt(1).widget().value())
        msg.pow_right = int(self.man_pow_r.itemAt(1).widget().value())
        self.label_status.setText(f"Status: Raw Command Sent")
        self.node.send_raw_command(msg)

    def create_slider(self, label_text, min_val, max_val, default_val):
        layout = QVBoxLayout()
        lbl_title = QLabel(f"{label_text}: {default_val}")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default_val)
        slider.valueChanged.connect(lambda val: lbl_title.setText(f"{label_text}: {val}"))
        layout.addWidget(lbl_title)
        layout.addWidget(slider)
        return layout, lbl_title


class RosGuiNode(Node):
    def __init__(self):
        super().__init__('gui_controller')
        
        # 1. AIトリガー用 (Auto Strategy)
        self.pub_trigger = self.create_publisher(Bool, '/serve_trigger', 10)
        
        # 2. テンプレート呼び出し用 (Auto Template) ★追加
        self.client_shoot_template = self.create_client(Shoot, 'shoot')
        
        # 3. 軌道計算機用 (Semi-Auto)
        self.client_shot = self.create_client(TargetShot, 'target_shot')
        
        # 4. 直接制御用 (Full Manual)
        self.pub_raw = self.create_publisher(ShotParams, 'shot_command', 10)

    def publish_trigger(self):
        msg = Bool()
        msg.data = True
        self.pub_trigger.publish(msg)

    def send_shoot_template(self, difficulty):
        """ /shoot サービスを呼び出す (テンプレート) """
        if not self.client_shoot_template.service_is_ready():
            self.get_logger().warn("Service '/shoot' not ready!")
            return
        
        req = Shoot.Request()
        req.difficulty = int(difficulty)
        # テンプレートモードなので、difficulty>0のみを使用し、他の座標は無視される前提
        
        future = self.client_shoot_template.call_async(req)
        future.add_done_callback(lambda f: self.get_logger().info("Template shot called"))

    def send_target_shot(self, x, y, roll, spin, speed):
        """ /target_shot サービスを呼び出す (座標指定) """
        if not self.client_shot.service_is_ready():
            self.get_logger().warn("Service '/target_shot' not ready!")
            return
        req = TargetShot.Request()
        req.target_x = float(x)
        req.target_y = float(y)
        req.height_z = 0.0
        req.roll_deg = float(roll)
        req.spin = int(spin)
        req.speed_mode = speed
        self.client_shot.call_async(req)

    def send_raw_command(self, msg):
        """ 直接Topic送信 """
        self.pub_raw.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    ros_node = RosGuiNode()
    app = QApplication(sys.argv)
    gui = PingPongGUI(ros_node)
    gui.show()

    timer = QTimer()
    timer.timeout.connect(lambda: rclpy.spin_once(ros_node, timeout_sec=0.0))
    timer.start(10)

    exit_code = app.exec_()
    timer.stop()
    ros_node.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)

if __name__ == '__main__':
    main()