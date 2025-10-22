# PingPongRobot
![WIP](https://img.shields.io/badge/status-開発中-yellow)

## 概要
- 年齢に応じて難易度が変わる卓球ロボ（予定）

## 必要パッケージ
- numpy (<2)
- opencv-python
- cv_bridge
- deepface
- tensorflow

## HowToUse
```(bash)
# ビルド&セットアップ
colcon build
source install/setup.bash

# ノード実行
ros2 run pingpong image_publisher
ros2 run pingpong age_estimation
```

## ノード構成
- **image_publisher**  
  /data/images内の画像を `/camera/image_raw` トピックに定期送信します。  
  実装: [`pingpong.image_publisher`](src/pingpong/pingpong/image_publisher.py)

- **age_estimation**  
  画像から顔年齢を推定し、`age` トピックに結果を送信します。  
  初回起動時はモデルのダウンロードが発生します。  
  実装: [`pingpong.age_estimation`](src/pingpong/pingpong/age_estimation.py)

- **sample_image_subscriber**  
  画像サイズを受信・表示します。  
  実装: [`pingpong.sample_image_subscriber`](src/pingpong/pingpong/sample_image_subscriber.py)

## 動作確認済み環境
- OS: Windows 11（WSL2: Ubuntu 24.04.1 LTS）
- ROS2: Jazzy

## その他
卓球ロボ用コード  
ros2のパッケージとして動作します  
説明はdocs内にあります  


