import rclpy
from rclpy.node import Node
import numpy as np
import message_filters
import tf2_ros
import tf2_geometry_msgs
import struct

from sensor_msgs.msg import PointCloud2, CameraInfo
from vision_msgs.msg import Detection2DArray, Detection2D
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped, Quaternion

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')
        self.get_logger().info("Sensor Fusion Node 시작됨")

        # TF Buffer and Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribers
        self.lidar_sub = message_filters.Subscriber(self, PointCloud2, '/lidar/points')
        self.left_cam_det_sub = message_filters.Subscriber(self, Detection2DArray, '/camera/left/cones_2d')
        self.right_cam_det_sub = message_filters.Subscriber(self, Detection2DArray, '/camera/right/cones_2d')
        self.left_cam_info_sub = message_filters.Subscriber(self, CameraInfo, '/camera/left/camera_info')
        self.right_cam_info_sub = message_filters.Subscriber(self, CameraInfo, '/camera/right/camera_info')

        # Time Synchronizer
        # ApproximateTimeSynchronizer를 사용하여 센서 데이터 동기화
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.lidar_sub, self.left_cam_det_sub, self.right_cam_det_sub, self.left_cam_info_sub, self.right_cam_info_sub],
            queue_size=10, slop=0.1)
        self.ts.registerCallback(self.fused_callback)

        # Publishers
        self.fused_cones_publisher = self.create_publisher(MarkerArray, '/fused_cones_3d', 10)

        # Camera Intrinsics (placeholder, will be updated from CameraInfo)
        self.left_cam_K = None
        self.right_cam_K = None
        self.left_cam_frame_id = 'left_camera_link'
        self.right_cam_frame_id = 'right_camera_link'

    def convert_pointcloud2_to_numpy(self, msg):
        """PointCloud2 메시지를 NumPy 배열 (x, y, z)로 변환합니다.
        """
        point_step = msg.point_step
        data = msg.data
        points = []
        for i in range(0, len(data), point_step):
            x, y, z = struct.unpack_from('fff', data, i)
            points.append([x, y, z])
        return np.array(points, dtype=np.float32)

    def get_transform(self, target_frame, source_frame, stamp):
        """TF 변환을 조회합니다.
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame, stamp, rclpy.duration.Duration(seconds=0.1))
            return transform
        except tf2_ros.TransformException as ex:
            self.get_logger().warn(f'Could not transform {source_frame} to {target_frame}: {ex}')
            return None

    def project_3d_to_2d(self, point_3d_camera, camera_K, image_width, image_height):
        """카메라 좌표계의 3D 포인트를 2D 이미지 좌표로 투영합니다.
        """
        if camera_K is None:
            return None

        fx = camera_K[0, 0]
        fy = camera_K[1, 1]
        cx = camera_K[0, 2]
        cy = camera_K[1, 2]

        # 3D 포인트의 Z값이 0이거나 음수이면 투영 불가 (카메라 뒤에 있거나 평면에 있음)
        if point_3d_camera[2] <= 0:
            return None

        u = fx * (point_3d_camera[0] / point_3d_camera[2]) + cx
        v = fy * (point_3d_camera[1] / point_3d_camera[2]) + cy

        # 이미지 범위 내에 있는지 확인
        if 0 <= u < image_width and 0 <= v < image_height:
            return int(u), int(v)
        else:
            return None

    def fused_callback(self, lidar_msg, left_det_msg, right_det_msg, left_info_msg, right_info_msg):
        """LiDAR, 카메라 감지, 카메라 정보 메시지가 동기화되어 호출되는 콜백 함수.
        """
        self.get_logger().info("Fused callback called.")

        # 카메라 내부 파라미터 업데이트
        self.left_cam_K = np.array(left_info_msg.k).reshape((3, 3))
        self.right_cam_K = np.array(right_info_msg.k).reshape((3, 3))
        left_image_width = left_info_msg.width
        left_image_height = left_info_msg.height
        right_image_width = right_info_msg.width
        right_image_height = right_info_msg.height

        # PointCloud2를 NumPy 배열로 변환
        lidar_points_xyz = self.convert_pointcloud2_to_numpy(lidar_msg)

        if lidar_points_xyz.shape[0] == 0:
            self.fused_cones_publisher.publish(MarkerArray())
            self.get_logger().warn("Received empty LiDAR point cloud.")
            return

        fused_cones_markers = MarkerArray()
        marker_id = 0

        # LiDAR 프레임에서 base_link 프레임으로의 변환
        lidar_to_base_transform = self.get_transform('base_link', lidar_msg.header.frame_id, lidar_msg.header.stamp)
        if lidar_to_base_transform is None:
            self.get_logger().warn("Failed to get transform from lidar_link to base_link.")
            return

        # 각 카메라의 base_link에 대한 변환
        left_cam_to_base_transform = self.get_transform('base_link', self.left_cam_frame_id, left_info_msg.header.stamp)
        right_cam_to_base_transform = self.get_transform('base_link', self.right_cam_frame_id, right_info_msg.header.stamp)

        if left_cam_to_base_transform is None or right_cam_to_base_transform is None:
            self.get_logger().warn("Failed to get transform from camera_link to base_link.")
            return

        # --- 융합 로직 --- (2D 감지 결과와 3D LiDAR 포인트 연관)

        # LiDAR 포인트를 base_link 프레임으로 변환
        transformed_lidar_points_base = []
        for p in lidar_points_xyz:
            point_stamped = tf2_geometry_msgs.PointStamped()
            point_stamped.header.frame_id = lidar_msg.header.frame_id
            point_stamped.header.stamp = lidar_msg.header.stamp
            point_stamped.point.x = p[0]
            point_stamped.point.y = p[1]
            point_stamped.point.z = p[2]
            transformed_point = tf2_geometry_msgs.do_transform_point(point_stamped, lidar_to_base_transform)
            transformed_lidar_points_base.append([transformed_point.point.x, transformed_point.point.y, transformed_point.point.z])
        transformed_lidar_points_base = np.array(transformed_lidar_points_base)

        # 좌측 카메라 감지 처리
        self.process_camera_detections(
            left_det_msg, transformed_lidar_points_base, self.left_cam_K, 
            left_cam_to_base_transform, left_image_width, left_image_height, 
            fused_cones_markers, marker_id, 'left')
        marker_id = len(fused_cones_markers.markers)

        # 우측 카메라 감지 처리
        self.process_camera_detections(
            right_det_msg, transformed_lidar_points_base, self.right_cam_K, 
            right_cam_to_base_transform, right_image_width, right_image_height, 
            fused_cones_markers, marker_id, 'right')
        marker_id = len(fused_cones_markers.markers)

        self.fused_cones_publisher.publish(fused_cones_markers)

    def process_camera_detections(self, det_msg, lidar_points_base, cam_K, cam_transform, img_width, img_height, fused_cones_markers, marker_id_start, camera_side):
        current_marker_id = marker_id_start
        for detection in det_msg.detections:
            bbox = detection.bbox
            # 2D 바운딩 박스 중심
            bbox_center_u = bbox.center.x
            bbox_center_v = bbox.center.y
            bbox_half_width = bbox.size_x / 2.0
            bbox_half_height = bbox.size.y / 2.0

            # 2D 바운딩 박스 범위
            bbox_min_u = bbox_center_u - bbox_half_width
            bbox_max_u = bbox_center_u + bbox_half_width
            bbox_min_v = bbox_center_v - bbox_half_height
            bbox_max_v = bbox_center_v + bbox_half_height

            # LiDAR 포인트를 카메라 좌표계로 변환
            lidar_points_camera = []
            for p_base in lidar_points_base:
                point_stamped_base = tf2_geometry_msgs.PointStamped()
                point_stamped_base.header.frame_id = 'base_link'
                point_stamped_base.header.stamp = det_msg.header.stamp
                point_stamped_base.point.x = p_base[0]
                point_stamped_base.point.y = p_base[1]
                point_stamped_base.point.z = p_base[2]
                
                try:
                    transformed_point_camera = tf2_geometry_msgs.do_transform_point(point_stamped_base, cam_transform)
                    lidar_points_camera.append([transformed_point_camera.point.x, transformed_point_camera.point.y, transformed_point_camera.point.z])
                except tf2_ros.TransformException as ex:
                    self.get_logger().warn(f'Could not transform point from base_link to {camera_side}_camera_link: {ex}')
                    continue
            lidar_points_camera = np.array(lidar_points_camera)

            # 2D 바운딩 박스 내에 투영되는 LiDAR 포인트 찾기
            points_in_bbox_3d = []
            for p_cam_3d, p_base_3d in zip(lidar_points_camera, lidar_points_base):
                projected_2d = self.project_3d_to_2d(p_cam_3d, cam_K, img_width, img_height)
                if projected_2d is not None:
                    u, v = projected_2d
                    if bbox_min_u <= u <= bbox_max_u and bbox_min_v <= v <= bbox_max_v:
                        points_in_bbox_3d.append(p_base_3d) # base_link 프레임의 3D 포인트 저장
            
            if len(points_in_bbox_3d) > 0:
                # 융합된 라바콘의 3D 위치 (평균)
                fused_cone_3d_point = np.mean(points_in_bbox_3d, axis=0)

                # 마커 생성
                marker = Marker()
                marker.header.frame_id = 'base_link'
                marker.header.stamp = det_msg.header.stamp
                marker.ns = f"fused_cones_{camera_side}"
                marker.id = current_marker_id
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = fused_cone_3d_point[0]
                marker.pose.position.y = fused_cone_3d_point[1]
                marker.pose.position.z = fused_cone_3d_point[2]
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.5
                marker.scale.y = 0.5
                marker.scale.z = 0.5
                marker.color.a = 1.0
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0 # Green for fused cones

                fused_cones_markers.markers.append(marker)
                current_marker_id += 1

def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
