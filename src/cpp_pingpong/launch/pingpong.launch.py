import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 設定ファイルのパスを取得
    # ※注: ビルド後に install フォルダにコピーされたパスを取得します
    config = os.path.join(
        get_package_share_directory('cpp_pingpong'),
        'config',
        'params.yaml'
    )

    return LaunchDescription([
        # C++ 軌道計算ノード
        Node(
            package='cpp_pingpong',
            executable='ballistics', # CMakeLists.txtで定義した名前
            name='ballistics',
            parameters=[config] # YAMLを渡す
        ),
        # Python 戦略ノード
        Node(
            package='py_pingpong',
            executable='strategy_server', # setup.pyで定義した名前
            name='strategy_server',
            parameters=[config] # 同じYAMLを渡す
        ),
        # Python 視覚ノード (Vision)
        Node(
            package='py_pingpong',
            executable='analyzer',
            name='vision_analyzer'
        ),
        # C++ コントローラ (Hardware Driver相当)
        Node(
            package='cpp_pingpong',
            executable='controller',
            name='controller'
        )
    ])