# PingPongRobot
![WIP](https://img.shields.io/badge/status-開発中-yellow)

## 概要
- 年齢に応じて難易度が変わる卓球サーブロボ（予定）
- モータへの指令値を送るまで

## 主な機能
- 画像から年齢・姿勢を推定
- 複数の戦略モード
- 弾道シミュレーションによる軌道計算
- GUIコントローラー

## パッケージリスト
### cpp_pingpong (C++)
射出制御とテンプレート管理
- `ballistics`: 物理演算ベースの弾道計算
- `controller`: プリセットテンプレートによる射出制御

### py_pingpong (Python)
ビジョン・戦略・GUI
- `img_publisher`: カメラ/動画をPublish
- `pose_estimator`: 姿勢推定（YOLOv8-Pose）
- `age_estimator`: 年齢推定（DeepFace）
- `strategy_server`: サーブ戦略決定
- `gui_controller`: GUI

### pingpong_msgs (Messages/Services)
カスタムメッセージ・サービス定義
- `ShotParams. msg`: 射出パラメータ（位置、角度、パワー、ボール供給）
- `TargetShot.srv`: ターゲット座標指定サービス
- `Shoot.srv`: テンプレート選択サービス

## Quick start
~~~(bash)
# 1. ビルド
colcon build
source install/setup.bash

# 2. ランチャーでまとめて起動
ros2 launch cpp_pingpong pingpong.launch.py

# 3. GUIは別ターミナルで
ros2 launch py_pingpong gui_controller
~~~

## 必要パッケージ（年齢推定・顔認識をする場合）
- numpy (<2)
- opencv-python
- cv_bridge
- deepface
- tensorflow

## 動作確認済み環境
- OS: Windows 11（WSL2: Ubuntu 24.04.1 LTS）
- ROS2: Jazzy

## その他
卓球ロボ用コード  
ros2のパッケージとして動作します  
説明はdocs内にあります  



