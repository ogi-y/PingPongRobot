from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    serve_calculator_node = Node(
        package='pingpong',
        executable='serve_calculator',
        name='serve_calculator',
        output='screen'
    )

    vision_processor_node = Node(
        package='pingpong',
        executable='vision_processor',
        name='vision_processor',
        output='screen'
    )

    controller_node = Node(
        package='pingpong',
        executable='controller',
        name='controller',
        output='screen'
    )

    trigger_node = Node(
        package='pingpong',
        executable='trigger',
        name='vision_trigger',
        output='screen'
    )

    return LaunchDescription([
        serve_calculator_node,
        vision_processor_node,
        controller_node,
        #trigger_node,
    ])
