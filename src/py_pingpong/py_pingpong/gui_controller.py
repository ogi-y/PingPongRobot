import sys
import rclpy
from rclpy.node import Node
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QSlider, QComboBox, QGroupBox, 
                             QTabWidget, QSpinBox, QCheckBox, QRadioButton, QButtonGroup)
from PyQt5.QtCore import Qt, QTimer
import random
from pingpong_msgs.srv import TargetShot
from pingpong_msgs.srv import Shoot
from pingpong_msgs.msg import ShotParams
from std_msgs.msg import Bool, String
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue

class PingPongGUI(QWidget):
    def __init__(self, ros_node):
        super().__init__()
        self.node = ros_node
        self.auto_timer = QTimer()
        self.auto_timer.timeout.connect(self.send_auto_trigger)
        self.template_timer = QTimer()
        self.template_timer.timeout.connect(self.fire_template_service)
        self.robot_x_slider_widget = None
        self.robot_y_slider_widget = None
        self.semi_x_widget = None
        self.semi_y_widget = None
        self.semi_roll_widget = None
        self.semi_spin_widget = None
        self.semi_speed_widget = None
        self.chk_auto_age = None
        self.spin_age = None
        self.init_ui()
        # ROS age listener registration (keeps GUI in sync with estimation topic)
        self.node.register_age_listener(self.on_age_estimated)

    def init_ui(self):
        self.setWindowTitle('Ping Pong Robot Commander')
        self.setGeometry(100, 100, 650, 750)
        main_layout = QVBoxLayout()
        tabs = QTabWidget()
        tabs.addTab(self.create_auto_tab(), "AUTO (Strategy)")
        tabs.addTab(self.create_template_tab(), "AUTO (Template)")
        tabs.addTab(self.create_semiauto_tab(), "SEMI-AUTO (Target)")
        tabs.addTab(self.create_manual_tab(), "FULL MANUAL (Motors)")
        main_layout.addWidget(tabs)
        self.label_status = QLabel("Status: Ready")
        self.label_status.setAlignment(Qt.AlignCenter)
        self.label_status.setStyleSheet("font-size: 14px; font-weight: bold; color: #333; padding: 10px;")
        main_layout.addWidget(self.label_status)

        self.setLayout(main_layout)

    # AUTO (Strategy) モード
    def create_auto_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        lbl = QLabel("AI Strategy Mode:  Uses Camera & Vision.\nRobot decides based on player position.")
        lbl.setStyleSheet("font-weight: bold; color: #4CAF50;")
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        # 【追加】戦略モード選択
        group_strategy = QGroupBox("Strategy Mode")
        layout_strategy = QVBoxLayout()
        
        self.combo_strategy = QComboBox()
        self.combo_strategy. addItems([
            'Adaptive (Smart AI - Default)',
            'Chase (Follow Player)',
            'Avoid (Opposite Side)',
            'Random (Unpredictable)',
            'Center (Fixed Center)',
            'Beginner (Super Easy)'
        ])
        self.combo_strategy.setCurrentIndex(0)
        self.combo_strategy.currentIndexChanged.connect(self.update_strategy_mode)
        
        layout_strategy.addWidget(QLabel("Select AI behavior:"))
        layout_strategy. addWidget(self.combo_strategy)
        group_strategy.setLayout(layout_strategy)
        layout.addWidget(group_strategy)

        # 手の設定
        group_hand = QGroupBox("Target Player Hand Setting")
        layout_hand = QHBoxLayout()
        
        self.rb_right = QRadioButton("Right Hand (Standard)")
        self.rb_left = QRadioButton("Left Hand")
        self.rb_right. setChecked(True)

        self.bg_hand = QButtonGroup()
        self.bg_hand.addButton(self.rb_right, 0)
        self.bg_hand.addButton(self.rb_left, 1)
        self.bg_hand.buttonClicked.connect(self.update_hand_setting)

        layout_hand.addWidget(self.rb_right)
        layout_hand.addWidget(self.rb_left)
        group_hand.setLayout(layout_hand)
        layout.addWidget(group_hand)

        # 間隔設定
        group_interval = QGroupBox("Firing Interval")
        layout_int = QHBoxLayout()
        layout_int.addWidget(QLabel("Interval (sec):"))
        self.spin_interval = QSpinBox()
        self.spin_interval.setRange(2, 10)
        self.spin_interval.setValue(3)
        layout_int.addWidget(self. spin_interval)
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

    def update_hand_setting(self):
        if self.rb_right.isChecked():
            hand = "right"
        else:
            hand = "left"
        
        self.label_status.setText(f"Status: Setting hand to {hand.upper()}...")
        self.node.set_vision_hand_param(hand)
    
    def update_strategy_mode(self):
        """戦略モードを変更"""
        strategy_map = {
            0: 'adaptive',
            1: 'chase',
            2: 'avoid',
            3: 'random',
            4: 'center',
            5: 'beginner'
        }
        
        mode = strategy_map. get(self.combo_strategy. currentIndex(), 'adaptive')
        self.label_status.setText(f"Status: Setting strategy to {mode. upper()}...")
        self.node.set_strategy_param(mode)

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

    # AUTO (Template) モード
    def create_template_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        lbl = QLabel("Template Mode: Controller's internal presets.\nSelect multiple levels to mix difficulties.")
        lbl.setStyleSheet("font-weight: bold; color: #9C27B0;")
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        group_auto = QGroupBox("Auto Level Select")
        layout_auto = QHBoxLayout()
        self.chk_auto_age = QCheckBox("Auto select by age")
        self.chk_auto_age.stateChanged.connect(self.on_auto_age_toggled)
        layout_auto.addWidget(self.chk_auto_age)

        layout_auto.addWidget(QLabel("Player age:"))
        self.spin_age = QSpinBox()
        self.spin_age.setRange(5, 99)
        self.spin_age.setValue(25)
        self.spin_age.valueChanged.connect(self.on_age_value_changed)
        layout_auto.addWidget(self.spin_age)
        layout_auto.addStretch()
        group_auto.setLayout(layout_auto)
        layout.addWidget(group_auto)

        # 難易度選択
        group_level = QGroupBox("Difficulty Levels (Mixable)")
        layout_level = QHBoxLayout()
        
        self.chk_lv1 = QCheckBox("Lv1: Easy")
        self.chk_lv2 = QCheckBox("Lv2: Normal")
        self.chk_lv3 = QCheckBox("Lv3: Hard")
        self.chk_lv4 = QCheckBox("Lv4: Pro")
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

    def on_auto_age_toggled(self):
        if self.chk_auto_age.isChecked():
            self.set_level_checkboxes_enabled(False)
            self.update_levels_from_age()
        else:
            self.set_level_checkboxes_enabled(True)
            self.label_status.setText("Status: Manual level selection")

    def on_age_value_changed(self, _value):
        if self.chk_auto_age.isChecked():
            self.update_levels_from_age()

    def on_age_estimated(self, age_value):
        """Age estimation callback from ROS topic"""
        # Bound the age to spin box limits then update UI; triggers auto-level refresh when enabled
        bounded = max(self.spin_age.minimum(), min(self.spin_age.maximum(), int(age_value)))
        if self.spin_age.value() != bounded:
            self.spin_age.setValue(bounded)
        elif self.chk_auto_age.isChecked():
            # If value unchanged but auto mode on, still refresh levels
            self.update_levels_from_age()

    def set_level_checkboxes_enabled(self, enabled):
        for chk in [self.chk_lv1, self.chk_lv2, self.chk_lv3, self.chk_lv4]:
            chk.setEnabled(enabled)

    def apply_level_checks(self, levels):
        self.chk_lv1.setChecked(1 in levels)
        self.chk_lv2.setChecked(2 in levels)
        self.chk_lv3.setChecked(3 in levels)
        self.chk_lv4.setChecked(4 in levels)

    def calculate_levels_for_age(self, age):
        # レベル調整
        if age < 10:
            return [1]
        if age < 20:
            return [1, 2]
        if age < 40:
            return [2, 3]
        if age < 50:
            return [3]
        if age < 80:
            return [3, 4]
        return [1, 2]

    def update_levels_from_age(self):
        age = self.spin_age.value()
        levels = self.calculate_levels_for_age(age)
        self.apply_level_checks(levels)
        level_text = ",".join([str(lv) for lv in levels])
        self.label_status.setText(f"Status: Auto level by age {age} -> Lv{level_text}")

    def collect_active_levels(self):
        if self.chk_auto_age and self.chk_auto_age.isChecked():
            levels = self.calculate_levels_for_age(self.spin_age.value())
            self.apply_level_checks(levels)
            return levels
        active_levels = []
        if self.chk_lv1.isChecked():
            active_levels.append(1)
        if self.chk_lv2.isChecked():
            active_levels.append(2)
        if self.chk_lv3.isChecked():
            active_levels.append(3)
        if self.chk_lv4.isChecked():
            active_levels.append(4)
        return active_levels

    def toggle_template_fire(self):
        if self.btn_tmpl_toggle.isChecked():
            active_levels = self.collect_active_levels()
            if not active_levels:
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
        active_levels = self.collect_active_levels()
        
        if not active_levels:
            self.label_status.setText("Status: No level selected!")
            return

        selected_difficulty = random.choice(active_levels)
        self.label_status.setText(f"Status: Calling /shoot (Diff:{selected_difficulty})")
        self.node.send_shoot_template(selected_difficulty)

    # SEMI-AUTO モード
    def create_semiauto_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        lbl = QLabel("Target Mode: Specify coordinates manually.")
        layout.addWidget(lbl)
        group_robot_pos = QGroupBox("Robot Position (Info)")
        layout_robot = QVBoxLayout()
        
        robot_x = self.node.get_parameter('robot_x_position').value
        robot_y = self.node.get_parameter('robot_y_position').value
        
        self.label_robot_pos = QLabel(f"Current Position: X={robot_x:.0f}mm, Y={robot_y:.0f}mm")
        self.label_robot_pos.setStyleSheet("font-weight: bold; color: #2196F3;")
        layout_robot.addWidget(self.label_robot_pos)
        
        # ボタンで位置を更新（オプション）
        btn_update_pos = QPushButton("Refresh Robot Position")
        btn_update_pos.clicked.connect(self.update_robot_position_display)
        layout_robot.addWidget(btn_update_pos)
        
        group_robot_pos.setLayout(layout_robot)
        layout.addWidget(group_robot_pos)

        group_robot_edit = QGroupBox("Robot Position Control (Advanced)")
        layout_robot_edit = QVBoxLayout()

        self.robot_x_slider, self.robot_x_slider_widget = self.create_slider("Robot X Position", 0, 1525, 762)
        layout_robot_edit.addLayout(self.robot_x_slider)

        self.robot_y_slider, self.robot_y_slider_widget = self.create_slider("Robot Y Position", -500, 500, 0)
        layout_robot_edit.addLayout(self.robot_y_slider)

        btn_set_robot_pos = QPushButton("Set Robot Position")
        btn_set_robot_pos.clicked.connect(self.set_robot_position)
        layout_robot_edit.addWidget(btn_set_robot_pos)

        group_robot_edit.setLayout(layout_robot_edit)
        layout.addWidget(group_robot_edit)

        self.semi_x, self.semi_x_widget = self.create_slider("Target X (Width)", 0, 1525, 762)
        layout.addLayout(self.semi_x)
        self.semi_y, self.semi_y_widget = self.create_slider("Target Y (Depth)", 0, 2740, 2200)
        layout.addLayout(self.semi_y)
        self.semi_roll, self.semi_roll_widget = self.create_slider("Head Roll (deg)", -45, 45, 0)
        layout.addLayout(self.semi_roll)
        
        self.semi_spin, self.semi_spin_widget = self.create_slider("Spin Power (Diff)", -50, 50, 0)
        layout.addLayout(self.semi_spin)

        self.semi_speed, self.semi_speed_widget = self.create_slider("Total Speed (L+R)", 0, 100, 15)
        layout.addLayout(self.semi_speed)

        btn_fire = QPushButton("FIRE TARGET SHOT")
        btn_fire.setFixedHeight(60)
        btn_fire.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        btn_fire.clicked.connect(self.fire_semiauto)
        layout.addWidget(btn_fire)

        layout.addStretch()
        tab.setLayout(layout)
        return tab
    def update_robot_position_display(self):
        """ロボット位置表示を更新"""
        robot_x = self.node.get_parameter('robot_x_position').value
        robot_y = self.node.get_parameter('robot_y_position').value
        self.label_robot_pos.setText(f"Current Position: X={robot_x:.0f}mm, Y={robot_y:.0f}mm")
        self.label_status.setText(f"Status: Robot at ({robot_x:.0f}, {robot_y:.0f})")

    def set_robot_position(self):
        """ロボット位置スライダーの値からロボット位置を設定"""
        # 【修正】保存したスライダーウィジェットから値を取得
        robot_x = self.robot_x_slider_widget.value()
        robot_y = self.robot_y_slider_widget.value()
        
        # RosGuiNodeのset_robot_positionメソッドを呼び出し
        self.node.set_robot_position(robot_x, robot_y)
        
        # 表示を更新
        self.update_robot_position_display()

    def fire_semiauto(self):
        x = float(self.semi_x_widget.value())
        y = float(self.semi_y_widget.value())
        roll = float(self.semi_roll_widget.value())
        spin = int(self.semi_spin_widget.value())
        
        # --- 【変更点】値をintで取得 ---
        speed = int(self.semi_speed_widget.value())
        
        self.label_status.setText(f"Status: Semi-Auto -> ({x:.0f}, {y:.0f}) Spd:{speed}")
        self.node.send_target_shot(x, y, roll, spin, speed)

    # FULL MANUAL モード
    def create_manual_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        lbl = QLabel("Direct Control:  Set raw values for all motors.")
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
        self.man_pow_r, _ = self. create_slider("Power Right (-100-100)", -100, 100, 0)
        layout.addLayout(self.man_pow_r)

        # 【追加】ボール供給トグルボタン（改良版）
        group_feed = QGroupBox("Ball Feeder Control")
        layout_feed = QHBoxLayout()
        
        lbl_feed = QLabel("Enable Ball Feed:")
        lbl_feed.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout_feed.addWidget(lbl_feed)
        
        self.btn_feed_toggle = QPushButton("OFF")
        self.btn_feed_toggle.setCheckable(True)
        self.btn_feed_toggle.setChecked(False)
        self.btn_feed_toggle.setFixedSize(100, 40)
        self.btn_feed_toggle.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                color: white;
                font-weight: bold;
                font-size: 18px;
                border-radius:  5px;
                border: 2px solid #C62828;
            }
            QPushButton:checked {
                background-color: #4CAF50;
                border:  2px solid #2E7D32;
            }
        """)
        self.btn_feed_toggle.clicked.connect(self.toggle_feed_ball)
        layout_feed.addWidget(self. btn_feed_toggle)
        
        layout_feed.addStretch()
        group_feed.setLayout(layout_feed)
        layout.addWidget(group_feed)

        btn_fire = QPushButton("FIRE RAW COMMAND")
        btn_fire.setFixedHeight(60)
        btn_fire.setStyleSheet("background-color: #FF9800; color:  white; font-weight: bold;")
        btn_fire.clicked.connect(self.fire_manual)
        layout.addWidget(btn_fire)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def toggle_feed_ball(self):
        """ボール供給ON/OFF切り替え"""
        if self.btn_feed_toggle.isChecked():
            self.btn_feed_toggle.setText("ON")
            self.label_status.setText("Status: Ball Feed ENABLED")
        else:
            self.btn_feed_toggle.setText("OFF")
            self.label_status.setText("Status: Ball Feed DISABLED")

    def fire_manual(self):
        msg = ShotParams()
        msg.pos = float(self. man_pos. itemAt(1).widget().value())
        msg.pitch_deg = float(self.man_pitch.itemAt(1).widget().value())
        msg.yaw_deg = float(self.man_yaw.itemAt(1).widget().value())
        msg.roll_deg = float(self.man_roll.itemAt(1).widget().value())
        msg.pow_left = int(self.man_pow_l. itemAt(1).widget().value())
        msg.pow_right = int(self.man_pow_r.itemAt(1).widget().value())
        
        # 【修正】トグルボタンの状態でfeed_ballを決定
        msg.feed_ball = 1 if self.btn_feed_toggle.isChecked() else 0
        
        feed_status = "ON" if msg.feed_ball == 1 else "OFF"
        self. label_status.setText(f"Status: Raw Command Sent (Feed:  {feed_status})")
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
        # 【修正】レイアウトとスライダーウィジェット両方を返す
        return layout, slider


class RosGuiNode(Node):
    def __init__(self):
        super().__init__('gui_controller')
        
        self.declare_parameter('robot_x_position', 762.5)
        self.declare_parameter('robot_y_position', 0.0)
        self.latest_age = None
        self.age_listener = None
        self.pub_trigger = self.create_publisher(Bool, '/serve_trigger', 10)
        self.client_shoot_template = self.create_client(Shoot, 'shoot')
        self.client_shot = self.create_client(TargetShot, 'target_shot')
        self.pub_raw = self.create_publisher(ShotParams, 'shot_command', 10)
        self.client_set_vision_param = self.create_client(SetParameters, '/pose_estimation_node/set_parameters')
        self.sub_age = self.create_subscription(String, '/age/estimation_result', self.age_callback, 10)

    # パラメータ設定送信
    def set_vision_hand_param(self, hand_value):
        if not self.client_set_vision_param.service_is_ready():
            self.get_logger().warn("Vision Analyzer param service not ready. Is the node running?")
            return

        req = SetParameters.Request()
        
        # パラメータオブジェクトの作成
        param = Parameter()
        param.name = "hand_side"  # VisionAnalyzer側で declare_parameter した名前
        param.value = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=hand_value)
        
        req.parameters = [param]
        
        future = self.client_set_vision_param.call_async(req)
        future.add_done_callback(lambda f: self.get_logger().info(f"Hand param updated to: {hand_value}"))
    
    def set_strategy_param(self, strategy_value):
        """strategy_serverのstrategy_modeパラメータを変更"""
        client = self.create_client(SetParameters, '/strategy_server/set_parameters')
        
        if not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Strategy Server param service not ready.")
            return

        req = SetParameters.Request()
        param = Parameter()
        param.name = "strategy_mode"
        param. value = ParameterValue(type=ParameterType. PARAMETER_STRING, string_value=strategy_value)
        req.parameters = [param]
        
        future = client.call_async(req)
        future.add_done_callback(lambda f: self.get_logger().info(f"Strategy mode updated to:  {strategy_value}"))

    def publish_trigger(self):
        msg = Bool()
        msg.data = True
        self.pub_trigger.publish(msg)

    def send_shoot_template(self, difficulty):
        if not self.client_shoot_template.service_is_ready():
            self.get_logger().warn("Service '/shoot' not ready!")
            return
        
        req = Shoot.Request()
        req.difficulty = int(difficulty)
        future = self.client_shoot_template.call_async(req)
        future.add_done_callback(lambda f: self.get_logger().info("Template shot called"))

    def send_target_shot(self, x, y, roll, spin, speed):
        if not self.client_shot.service_is_ready():
            self.get_logger().warn("Service '/target_shot' not ready!")
            return
        req = TargetShot.Request()
        req.robot_x = float(self.get_parameter('robot_x_position').value)
        req.robot_y = float(self.get_parameter('robot_y_position').value)
        req.target_x = float(x)
        req.target_y = float(y)
        req.height_z = 0.0
        req.roll_deg = float(roll)
        req.spin = int(spin)
        req.speed = int(speed)
        self.client_shot.call_async(req)

    def send_raw_command(self, msg):
        self.pub_raw.publish(msg)
    
    def set_robot_position(self, x, y):
        """ロボット位置パラメータを動的に更新"""
        client = self.create_client(SetParameters, '/gui_controller/set_parameters')
        
        if not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Cannot set robot position")
            return
        
        req = SetParameters.Request()
        
        param_x = Parameter()
        param_x.name = "robot_x_position"
        param_x.value = ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(x))
        
        param_y = Parameter()
        param_y.name = "robot_y_position"
        param_y.value = ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(y))
        
        req.parameters = [param_x, param_y]
        
        future = client.call_async(req)
        future.add_done_callback(lambda f: self.get_logger().info(f"Robot position updated to: ({x}, {y})"))

    # --- Age estimation ---
    def register_age_listener(self, callback):
        self.age_listener = callback
        if self.latest_age is not None and self.age_listener:
            self.age_listener(self.latest_age)

    def age_callback(self, msg: String):
        try:
            age_val = int(float(msg.data))
        except (ValueError, TypeError):
            self.get_logger().warn(f"Invalid age message: {msg.data}")
            return
        self.latest_age = age_val
        if self.age_listener:
            self.age_listener(age_val)

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