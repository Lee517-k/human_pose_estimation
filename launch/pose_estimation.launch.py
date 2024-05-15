from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='human_pose_estimation',
            executable='pose_estimation',
            name='pose_estimation',
            output='screen',
            parameters=[]
        )
    ])
