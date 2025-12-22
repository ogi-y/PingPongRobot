import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    camera_source_arg = DeclareLaunchArgument(
        'cam_source',
        default_value="'0'",
        description='Camera source:  "0" (camera index), "/dev/video0" (device), or "./test.mp4" (video file)'
    )
    loop_video_arg = DeclareLaunchArgument(
        'loop_video',
        default_value='true',
        description='Loop video playback (only for video files)'
    )
    strategy_mode_arg = DeclareLaunchArgument(
        'strategy_mode',
        default_value='adaptive',
        description='Strategy mode: adaptive, chase, avoid, random, center, or beginner'
    )
    hand_side_arg = DeclareLaunchArgument(
        'hand_side',
        default_value='right',
        description='Target hand side: right or left'
    )

    cam_source = LaunchConfiguration('cam_source')
    loop_video = LaunchConfiguration('loop_video')
    strategy_mode = LaunchConfiguration('strategy_mode')
    hand_side = LaunchConfiguration('hand_side')

    return LaunchDescription([
        camera_source_arg,
        loop_video_arg,
        strategy_mode_arg,
        hand_side_arg,

        Node(
            package='py_pingpong',
            executable='img_publisher',
            name='img_publisher',
            parameters=[{
                'cam_source': cam_source,
                'fps':  30,
                'width':  640,
                'height':  480,
                'loop_video': loop_video,
                'use_v4l2': True
            }],
            output='screen'
        ),

        Node(
            package='py_pingpong',
            executable='pose_estimator',
            name='pose_estimator',
            parameters=[{
                'hand_side':  hand_side,
                'model_path': 'yolov8n-pose.pt',
                'confidence_threshold': 0.5,
                'process_interval': 1
            }],
            output='screen'
        ),

        Node(
            package='py_pingpong',
            executable='age_estimator',
            name='age_estimator',
            parameters=[{
                'detector_backend': 'opencv',
                'process_interval': 120,
                'enforce_detection': False
            }],
            output='screen'
        ),

        Node(
            package='py_pingpong',
            executable='strategy_server',
            name='strategy_server',
            parameters=[{
                'court_width': 1525.0,
                'court_length': 2740.0,
                'strategy_mode': strategy_mode,
                'robot_x_position': 762.5,
                'robot_y_position': 0.0
            }],
            output='screen'
        ),

        Node(
            package='cpp_pingpong',
            executable='ballistics',
            name='ballistics',
            parameters=[{
                'robot_x_position': 762.5,
                'robot_y_position': 0.0
            }],
            output='screen'
        ),

        Node(
            package='cpp_pingpong',
            executable='controller',
            name='controller',
            output='screen'
        ),
    ])