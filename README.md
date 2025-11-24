# PingPongRobot
![WIP](https://img.shields.io/badge/status-開発中-yellow)

## 概要
- 年齢に応じて難易度が変わる卓球サーブロボ（予定）
- ROS2パッケージ部分だけ抜き出したものを後日作成予定
- 3つのパッケージが含まれています

## パッケージリスト
- cpp_pingpong:モーター制御用メッセージの生成
- pingpong_msgs:モーター制御用カスタムメッセージ
- pingpong:顔認識、年齢推定、射出軌道の計算など（必須ではない）

## HowToUse
1. コントローラーを起動
~~~(bash)
ros2 run cpp_pingpong controller 
~~~
2. 発射サービスを呼び出し
~~~(bash)
ros2 service call /shoot pingpong_msgs/srv/Shoot "{difficulty: 1}"
~~~

## 発射サービスの説明
- マニュアルモードはdifficulty: 0を指定
~~~(bash)
ros2 service call /shoot pingpong_msgs/srv/Shoot "{difficulty: 1, pos: 1000.0, pow_right: 10, pow_left: 10}"
~~~
- difficultyが0以外のときはテンプレートから自動選択
- 入力しなかったパラメータは0として扱う

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



