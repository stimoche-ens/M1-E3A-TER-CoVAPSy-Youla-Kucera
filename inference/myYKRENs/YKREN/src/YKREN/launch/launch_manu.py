from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.event_handlers import OnProcessStart
import os

def generate_launch_description():
    # 2. Define entities
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen',
        parameters=[{
            # Forces joy_node to use the default Bluetooth mapping
            'dev': '/dev/input/js0',
            'deadzone': 0.05,
            'autorepeat_rate': 20.0,
        }]
    )

    cmdR_node = Node(
        package='YKREN',
        executable='CoVAPSy_cmdR',
        name='cmdR_node',
        output='screen'
    )

    conduiteM_node = Node(
        package='YKREN',
        executable='CoVAPSy_conduiteM',
        name='conduiteM_node',
        output='screen'
    )

    # 3. Define Sequential Event Handlers
    # Trigger conduiteR_node only after cmdR_node has started
    start_conduiteM_handler = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=cmdR_node,
            on_start=[conduiteM_node]
        )
    )

    # 4. Construct the initial execution graph (only the first node is launched immediately)
    return LaunchDescription([
        cmdR_node,
        joy_node,
        start_conduiteM_handler,
    ])
