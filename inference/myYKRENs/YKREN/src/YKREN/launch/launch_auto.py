from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.event_handlers import OnProcessStart
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 1. Resolve external package paths
    rplidar_pkg_dir = get_package_share_directory('rplidar_ros')
    lidar_launch_path = os.path.join(rplidar_pkg_dir, 'launch', 'rplidar_s2_launch.py')

    # 2. Define entities
    cmdR_node = Node(
        package='YKREN',
        executable='CoVAPSy_cmdR',
        name='cmdR_node',
        output='screen'
    )

    conduiteR_node = Node(
        package='YKREN',
        executable='CoVAPSy_conduiteR',
        name='conduiteR_node',
        output='screen'
    )

    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(lidar_launch_path)
    )

    # 3. Define Sequential Event Handlers
    # Trigger conduiteR_node only after cmdR_node has started
    start_conduiteR_handler = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=cmdR_node,
            on_start=[conduiteR_node]
        )
    )

    # Trigger lidar_launch only after conduiteR_node has started
    start_lidar_handler = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=conduiteR_node,
            on_start=[lidar_launch]
        )
    )

    # 4. Construct the initial execution graph (only the first node is launched immediately)
    return LaunchDescription([
        cmdR_node,
        start_conduiteR_handler,
        start_lidar_handler
    ])
