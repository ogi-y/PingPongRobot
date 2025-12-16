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

    # カメラソース指定引数
    camera_source_arg = DeclareLaunchArgument(
        'source',          # 実行時に指定する名前
        default_value='0', # 指定なしの場合の値
        description='Camera device ID or Video URL for vision_main'
    )

    source_config = LaunchConfiguration('source')

    return LaunchDescription([
        camera_source_arg,

        # 1. C++ 軌道計算ノード
        Node(
            package='cpp_pingpong',
            executable='ballistics',
            name='ballistics',
            parameters=[config],
            output='screen'
        ),
        
        # 2. Python 戦略ノード
        Node(
            package='py_pingpong',
            executable='strategy_server',
            name='strategy_server',
            parameters=[config],
            output='screen'
        ),
        
        # 3. Python 視覚ノード (メイン: YOLO/カメラ)
        # ※ここに source パラメータを渡します
        Node(
            package='py_pingpong',
            executable='vision_main',
            name='vision_main',
            parameters=[{
                'source': source_config
            }],
            output='screen'
        ),

        # 4. Python 視覚ノード (サブ: 年齢推定)
        # ※独立して起動させます
        Node(
            package='py_pingpong',
            executable='vision_age',
            name='vision_age',
            output='screen'
        ),

        # 5. C++ コントローラ
        Node(
            package='cpp_pingpong',
            executable='controller',
            name='controller',
            output='screen'
        )
    ])