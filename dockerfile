FROM ros:humble

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    ros-humble-cv-bridge \
    ros-humble-vision-opencv \
    ros-humble-image-transport \
    git wget unzip \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージのインストール
RUN pip3 install --no-cache-dir opencv-python pillow
RUN pip install ultralytics

# ROS環境を自動的にセットアップ
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

# ワークスペース設定
WORKDIR /root/workspace

# シェル起動時のデフォルトコマンド
CMD ["/bin/bash"]