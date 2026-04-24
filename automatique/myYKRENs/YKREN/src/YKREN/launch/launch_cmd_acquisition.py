from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
        ),
	Node(
            package='YKREN',
            executable='CoVAPSy_conduiteM',
            name='CoVAPSy_conduiteM',
        ),
	Node(
            package='YKREN',
            executable='CoVAPSy_cmdR',
            name='CoVAPSy_cmdR',
        ),
	Node(
            package='rplidar_ros',
            executable='rplidar_node',
            name='rplidar_node',
        )     
 ])
