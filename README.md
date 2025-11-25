# PingPongRobot
![WIP](https://img.shields.io/badge/status-開発中-yellow)

## 概要
- 年齢に応じて難易度が変わる卓球サーブロボ（予定）
- ROS2パッケージ部分だけ抜き出したものを後日作成予定
- 3つのパッケージが含まれています

## パッケージリスト
- cpp_pingpong:モーター制御用メッセージの生成
- pingpong_msgs:モーター制御用カスタムメッセージ
- py_pingpong:画像処理や軌道計算など

- pingpong:顔認識、年齢推定、射出軌道の計算など（未使用）

## HowToUse
1. ノード起動
~~~(bash)
ros2 launch cpp_pingpong pingpong.launch.py
~~~
2. GUIの起動
~~~(bash)
ros2 run py_pingpong gui_controller 
~~~
3. 発射コマンドの確認
~~~(bash)
ros2 topic echo /shot_command
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



