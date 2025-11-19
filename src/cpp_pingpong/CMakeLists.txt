cmake_minimum_required(VERSION 3.8)
project(cpp_pingpong)

# C++17を使用
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 1. 依存パッケージの検索
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# 2. 実行ファイルの定義
add_executable(vision_processor src/vision_processor.cpp)

# 3. インクルードディレクトリの設定
# OpenCVのヘッダ場所を教える
target_include_directories(vision_processor PUBLIC
  ${OpenCV_INCLUDE_DIRS}
)

# 4. ライブラリのリンク設定
# OpenCVのライブラリファイル自体をリンクする
target_link_libraries(vision_processor
  ${OpenCV_LIBS}
)

# 5. ROS 2 依存関係の設定 (ament)
# ここで rclcpp, cv_bridge などのヘッダ場所とライブラリリンクを一括で行います
ament_target_dependencies(vision_processor
  rclcpp
  std_msgs
  sensor_msgs
  cv_bridge
)

# 6. インストール設定
install(TARGETS vision_processor
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()