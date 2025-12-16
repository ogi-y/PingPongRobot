import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # 設定ファイルのパスを取得
    config = os.path.join(
        get_package_share_directory('cpp_pingpong'),
        'config',
        'params.yaml'
    )

    camera_source_arg = DeclareLaunchArgument(
        'camera_source',          # 実行時に指定する名前
        default_value='0',        # 何も指定しなかった場合の値
        description='Camera device ID or Video URL for vision_analyzer'
    )

    source_config = LaunchConfiguration('source')

    return LaunchDescription([
        camera_source_arg,

        # C++ 軌道計算ノード
        Node(
            package='cpp_pingpong',
            executable='ballistics',
            name='ballistics',
            parameters=[config]
        ),
        # Python 戦略ノード
        Node(
            package='py_pingpong',
            executable='strategy_server',
            name='strategy_server',
            parameters=[config]
        ),
        # Python 視覚ノード (Vision)
        Node(
            package='py_pingpong',
            executable='analyzer',
            name='vision_analyzer',
            parameters=[{
                'source': source_config
            }]
        ),
        # C++ コントローラ
        Node(
            package='cpp_pingpong',
            executable='controller',
            name='controller'
        )
    ])