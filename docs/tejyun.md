# 起動手順
1. 初回　docker compose up --build
    二回目以降 docker compose up
2. docker exec -it ros2_dev bash

# 再起動手順
1. docker compose down
2. docker compose up --build
3. docker exec -it ros2_dev bash

DeepLearning系のパッケージを毎回インストールする手間を考えるとDockerをつかうよりubuntuにros2を入れてしまうほうが楽では？