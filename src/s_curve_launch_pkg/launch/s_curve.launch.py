import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import math # Added import

def generate_launch_description():
    # Get the share directory for our package
    s_curve_launch_pkg_share = get_package_share_directory('s_curve_launch_pkg')

    # --- Nodes ---

    # 1. Lidar Sensor Node
    lidar_sensor_node = Node(
        package='lidar_pkg',
        executable='lidar_sensor_node',
        name='lidar_sensor_node',
        output='screen'
    )

    # 2. Sensor Fusion Node
    sensor_fusion_node = Node(
        package='sensor_fusion_pkg',
        executable='sensor_fusion_node',
        name='sensor_fusion_node',
        output='screen',
        remappings=[
            ('/lidar/points', '/lidar/points'),
            ('/camera/left/cones_2d', '/camera/left/cones_2d'),
            ('/camera/right/cones_2d', '/camera/right/cones_2d'),
            ('/camera/left/camera_info', '/camera/left/camera_info'),
            ('/camera/right/camera_info', '/camera/right/camera_info')
        ]
    )

    # 3. Path Generation Node
    path_generation_node = Node(
        package='path_generation_pkg',
        executable='path_generation_node',
        name='path_generation_node',
        output='screen',
        remappings=[
            ('/detected_cones', '/fused_cones_3d') # Path generation now uses fused 3D cones
        ]
    )

    # 4. Control Node
    control_node = Node(
        package='control_pkg',
        executable='control_node',
        name='control_node',
        output='screen'
    )

    # 5. ERP42 Driver Node
    erp42_driver_node = Node(
        package='erp42_driver_pkg',
        executable='erp42_driver_node',
        name='erp42_driver_node',
        output='screen'
    )

    # 6. EKF Node for Localization
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[os.path.join(s_curve_launch_pkg_share, 'config', 'ekf.yaml')],
        remappings=[
            ('odom', 'odom'), # EKF subscribes to /odom (from erp42_driver_pkg)
            ('odometry/filtered', 'odometry/filtered') # EKF publishes to /odometry/filtered (for ControlNode)
        ]
    )

    # 7. Robot State Publisher (for base_link to lidar_link TF)
    static_transform_publisher_base_lidar = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher_base_lidar',
        arguments=['0.6', '0', '0.8', '0', '0', '0', 'base_link', 'lidar_link'] # Lidar at 0.8m height, 0.6m forward from base_link
    )

    # 8. Left Camera Driver Node (Example: usb_cam)
    left_camera_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='left_camera_node',
        output='screen',
        parameters=[
            {'video_device': '/dev/video0'},
            {'image_width': 640},
            {'image_height': 480},
            {'pixel_format': 'yuyv'},
            {'camera_frame_id': 'left_camera_link'},
            {'camera_name': 'left_camera'},
            {'framerate': 30.0}
        ],
        remappings=[
            ('image_raw', 'camera/left/image_raw'),
            ('camera_info', 'camera/left/camera_info')
        ]
    )

    # 9. Right Camera Driver Node (Example: usb_cam)
    right_camera_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='right_camera_node',
        output='screen',
        parameters=[
            {'video_device': '/dev/video1'},
            {'image_width': 640},
            {'image_height': 480},
            {'pixel_format': 'yuyv'},
            {'camera_frame_id': 'right_camera_link'},
            {'camera_name': 'right_camera'},
            {'framerate': 30.0}
        ],
        remappings=[
            ('image_raw', 'camera/right/image_raw'),
            ('camera_info', 'camera/right/camera_info')
        ]
    )

    # 10. Static Transform Publisher for Left Camera
    static_transform_publisher_left_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher_left_camera',
        arguments=['0.6', '0.2', '1.1', '0', '0', str(math.radians(45)), 'base_link', 'left_camera_link'] # x, y, z, roll, pitch, yaw
    )

    # 11. Static Transform Publisher for Right Camera
    static_transform_publisher_right_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher_right_camera',
        arguments=['0.6', '-0.2', '1.1', '0', '0', str(math.radians(-45)), 'base_link', 'right_camera_link'] # x, y, z, roll, pitch, yaw
    )

    # 12. Cone Detector DL Node (for Left Camera)
    cone_detector_node_left = Node(
        package='cone_detector_dl_pkg',
        executable='cone_detector_node',
        name='cone_detector_node_left',
        output='screen',
        parameters=[
            {'image_topic_left': '/camera/left/image_raw'},
            {'detection_topic_left': '/camera/left/cones_2d'},
            {'model_path': 'models/best.pt'} # Relative to package share directory
        ],
        remappings=[
            ('image_raw', 'camera/left/image_raw'),
            ('detections', 'camera/left/cones_2d')
        ]
    )

    # 13. Cone Detector DL Node (for Right Camera)
    cone_detector_node_right = Node(
        package='cone_detector_dl_pkg',
        executable='cone_detector_node',
        name='cone_detector_node_right',
        output='screen',
        parameters=[
            {'image_topic_left': '/camera/right/image_raw'},
            {'detection_topic_left': '/camera/right/cones_2d'},
            {'model_path': 'models/best.pt'} # Relative to package share directory
        ],
        remappings=[
            ('image_raw', 'camera/right/image_raw'),
            ('detections', 'camera/right/cones_2d')
        ]
    )

    return LaunchDescription([
        lidar_sensor_node,
        # perception_node, # Removed as sensor_fusion_node takes over
        sensor_fusion_node,
        path_generation_node,
        control_node,
        erp42_driver_node,
        ekf_node,
        static_transform_publisher_base_lidar,
        left_camera_node,
        right_camera_node,
        static_transform_publisher_left_camera,
        static_transform_publisher_right_camera,
        cone_detector_node_left,
        cone_detector_node_right
    ])